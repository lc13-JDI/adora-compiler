// Prove the cgeist recurrence live' = live && mode[iv] >= 0 before replacing
// it with eight fixed slots. The state is local to one launch and is unobserved
// after the loop. The later checked ABI ensures all four objects are disjoint.
static LogicalResult normalizePrefixExit(scf::ForOp old, ArrayRef<Object> objects,
                                         Value &modeSample) {
  auto reject = [&](StringRef reason) -> LogicalResult {
    return old.emitOpError("C-entry: prefix exit ") << reason;
  };
  if (objects.size() != 4 || old.getNumResults() != 1 || !old->use_empty() ||
      !old.getResult(0).getType().isInteger(1) ||
      constant(old.getInitArgs()[0]).value_or(0) == 0 ||
      constant(old.getLowerBound()) != 0 || constant(old.getUpperBound()) != 8 ||
      constant(old.getStep()) != 1)
    return reject("requires one initially true bit, exactly eight slots and no live-out");

  auto yield = old.getBody()->getTerminator();
  auto active = yield->getOperand(0).getDefiningOp<scf::IfOp>();
  if (!active || active.getNumResults() != 1 || active->getParentOp() != old ||
      active.getCondition() != old.getRegionIterArgs()[0] ||
      active.getElseRegion().empty() || constant(active.elseYield().getOperand(0)) != 0)
    return reject("requires the carried bit to guard an absorbing false path");
  Value keep = active.thenYield().getOperand(0);
  auto sign = keep.getDefiningOp<arith::CmpIOp>();
  using P = arith::CmpIPredicate;
  if (!sign || sign.getPredicate() != P::sge || constant(sign.getRhs()) != 0)
    return reject("requires signed mode >= 0 as the next active bit");

  Value iv = old.getInductionVar();
  auto role = [&](Value memref, StringRef dtype, bool inout) {
    auto arg = dyn_cast<BlockArgument>(memref);
    if (!arg || arg.getOwner() != old->getBlock()) return false;
    const Object &obj = objects[arg.getArgNumber()];
    return obj.elements == 8 && obj.dtype == dtype && obj.partial == inout && obj.write == inout;
  };
  auto slotRead = [&](Value operand) -> memref::LoadOp {
    auto load = signedLeaf(operand).getDefiningOp<memref::LoadOp>();
    if (!load || load.getIndices().size() != 1 || load.getIndices()[0] != iv ||
        !role(load.getMemref(), "int16", false))
      return {};
    return load;
  };
  auto mode = slotRead(sign.getLhs());
  if (!mode || mode->getBlock() != &active.getThenRegion().front())
    return reject("requires the current signed mode slot");

  SmallVector<memref::StoreOp> stores;
  old.walk([&](memref::StoreOp store) { stores.push_back(store); });
  if (stores.size() != 1) return reject("requires exactly one destination write");
  auto store = stores.front();
  auto sample = store.getValue().getDefiningOp<memref::LoadOp>();
  auto guard = dyn_cast<scf::IfOp>(store->getParentOp());
  if (!sample || !guard || !guard.getElseRegion().empty() || guard.getNumResults() ||
      sample->getBlock() != &guard.getThenRegion().front() ||
      store.getIndices().size() != 1 || store.getIndices()[0] != iv ||
      !role(store.getMemref(), "uint16", true) || !role(sample.getMemref(), "uint16", false))
    return reject("requires one guarded uint16 source read and same-slot inout write");
  auto proceed = dyn_cast<scf::IfOp>(guard->getParentOp());
  if (!proceed || !proceed.getElseRegion().empty() || proceed.getNumResults() ||
      proceed.getCondition() != keep || proceed->getBlock() != &active.getThenRegion().front())
    return reject("requires the same signed exit decision to dominate the write");
  auto address = sample.getIndices()[0].getDefiningOp<arith::IndexCastOp>();
  auto index = address ? slotRead(address.getIn()) : memref::LoadOp();
  if (!index || index.getMemref() == mode.getMemref())
    return reject("requires a separate signed current-slot index");

  // Prove all and only the three predicates on the original indirect read.
  // In particular, a zero mode skips just this slot; it is not an exit.
  SmallVector<Value> terms{guard.getCondition()};
  unsigned matched = 0, conjunctions = 0;
  memref::LoadOp gateMode;
  while (!terms.empty()) {
    Value term = terms.pop_back_val();
    if (auto both = term.getDefiningOp<arith::AndIOp>()) {
      if (++conjunctions > 2) return reject("has an unsupported write guard");
      terms.push_back(both.getLhs());
      terms.push_back(both.getRhs());
      continue;
    }
    auto cmp = term.getDefiningOp<arith::CmpIOp>();
    if (!cmp) return reject("requires a conjunctive mode and index guard");
    auto leaf = slotRead(cmp.getLhs());
    auto c = constant(cmp.getRhs());
    if (leaf && leaf.getMemref() == mode.getMemref() && cmp.getPredicate() == P::ne && c == 0) {
      gateMode = leaf;
      matched |= 1;
    } else if (leaf && leaf == index && cmp.getPredicate() == P::sge && c == 0) {
      matched |= 2;
    } else if (leaf && leaf == index && cmp.getPredicate() == P::slt && c == 8) {
      matched |= 4;
    } else return reject("requires the same mode != 0 and the true index bounds [0,8)");
  }
  if (matched != 7) return reject("requires the full current-mode and index guard");

  // Every operation in the region to be erased must be accounted for. Reads
  // other than the two checked direct slots and the guarded sample might trap.
  auto checked = old.walk([&](Operation *op) {
    if (isa<arith::ConstantOp, arith::ExtSIOp, arith::CmpIOp, arith::AndIOp,
            arith::IndexCastOp>(op)) return WalkResult::advance();
    if (!op->getAttrs().empty()) return WalkResult::interrupt();
    if (op == old || op == active || op == proceed || op == guard || op == mode ||
        op == gateMode || op == index || op == sample || op == store || isa<scf::YieldOp>(op))
      return WalkResult::advance();
    return WalkResult::interrupt();
  });
  if (checked.wasInterrupted())
    return reject("contains unsupported operations, memory effects or attributes");

  // These direct reads are safe to prefetch at every slot; the indirect load
  // stays under its original three guards. Remember the mode SSA value so the
  // caller can attach the proven recurrence after validating the ordinary IR.
  OpBuilder b(old);
  auto loop = b.create<scf::ForOp>(old.getLoc(), old.getLowerBound(),
                                     old.getUpperBound(), old.getStep());
  b.setInsertionPointToStart(loop.getBody());
  Value lane = loop.getInductionVar();
  Value idx = b.create<memref::LoadOp>(index.getLoc(), index.getMemref(), ValueRange{lane});
  modeSample = b.create<memref::LoadOp>(mode.getLoc(), mode.getMemref(), ValueRange{lane});
  Value zero = b.create<arith::ConstantIntOp>(sign.getLoc(), 0, 16);
  Value eight = b.create<arith::ConstantIntOp>(guard.getLoc(), 8, 16);
  Value nonzero = b.create<arith::CmpIOp>(guard.getLoc(), P::ne, modeSample, zero);
  Value lower = b.create<arith::CmpIOp>(guard.getLoc(), P::sge, idx, zero);
  Value upper = b.create<arith::CmpIOp>(guard.getLoc(), P::slt, idx, eight);
  Value bounds = b.create<arith::AndIOp>(guard.getLoc(), lower, upper);
  Value ordinary = b.create<arith::AndIOp>(guard.getLoc(), nonzero, bounds);
  auto safe = b.create<scf::IfOp>(guard.getLoc(), ordinary, false);
  b.setInsertionPointToStart(&safe.getThenRegion().front());
  Value subscript = b.create<arith::IndexCastOp>(address.getLoc(), b.getIndexType(), idx);
  Value data = b.create<memref::LoadOp>(sample.getLoc(), sample.getMemref(), ValueRange{subscript});
  b.create<memref::StoreOp>(store.getLoc(), data, store.getMemref(), ValueRange{lane});
  old.erase();
  return success();
}
