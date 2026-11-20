// Merge just the effect region below the nearest fork separating two stores.
// The enclosing guard and its dynamic source load remain in their original
// scope. The caller then proves the single-write ABI, source bounds and types.
static LogicalResult mergeExclusiveWrites(Operation *loop, Value iv,
    ArrayRef<Object> objects, Value &selectedData, DenseSet<Operation *> &generated) {
  SmallVector<memref::StoreOp> stores;
  loop->walk([&](memref::StoreOp store) { stores.push_back(store); });
  if (stores.size() != 2) return success();
  auto reject = [](Operation *op, StringRef reason) -> LogicalResult {
    return op->emitError("C-entry: ") << reason;
  };
  auto left = stores[0], right = stores[1];
  auto destination = dyn_cast<BlockArgument>(left.getMemref());
  if (!destination || destination.getOwner() != loop->getBlock() ||
      !objects[destination.getArgNumber()].partial ||
      left.getMemref() != right.getMemref() ||
      left.getIndices().size() != 1 || right.getIndices().size() != 1 ||
      left.getIndices()[0] != iv || right.getIndices()[0] != iv)
    return reject(left, "conditional writes must address the same iteration index of one inout object");

  scf::IfOp split;
  for (Operation *op = left->getParentOp(); op != loop; op = op->getParentOp()) {
    auto branch = dyn_cast<scf::IfOp>(op);
    if (!branch || branch.getElseRegion().empty()) continue;
    Region &yes = branch.getThenRegion(), &no = branch.getElseRegion();
    if (yes.findAncestorOpInRegion(*left) && no.findAncestorOpInRegion(*right)) {
      split = branch;
      break;
    }
    if (yes.findAncestorOpInRegion(*right) && no.findAncestorOpInRegion(*left)) {
      split = branch;
      std::swap(left, right);
      break;
    }
  }
  if (!split)
    return reject(left, "one conditional write requires structurally mutually exclusive paths");

  // Check *every* operation before erasing either arm. Only already available
  // scalar values, total integer operations, and identical dominating reads
  // from a checked read-only argument can be evaluated outside their arm.
  SmallVector<Operation *> pure;
  SmallVector<std::pair<memref::LoadOp, Value>> forwarded;
  auto checked = split.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      if (!branch.getNumResults() && op->getAttrs().empty()) return WalkResult::advance();
    } else if (isa<scf::YieldOp>(op)) {
      if (!op->getNumOperands() && op->getAttrs().empty()) return WalkResult::advance();
    } else if (isa<memref::StoreOp>(op)) {
      if (op->getAttrs().empty()) return WalkResult::advance();
    } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
      auto arg = dyn_cast<BlockArgument>(load.getMemref());
      if (arg && arg.getOwner() == loop->getBlock() &&
          objects[arg.getArgNumber()].write)
        return WalkResult(reject(op, "explicit output reads in a write region are unsupported"));
      if (arg && arg.getOwner() == loop->getBlock() && load.getIndices().size() == 1 &&
          load.getIndices()[0] == iv && op->getAttrs().empty()) {
        Block *body = cast<BlockArgument>(iv).getOwner();
        Operation *root = body->findAncestorOpInBlock(*split);
        for (Operation &before : body->getOperations()) {
          if (&before == root) break;
          auto prior = dyn_cast<memref::LoadOp>(before);
          if (prior && prior.getMemref() == arg && prior.getIndices().size() == 1 &&
              prior.getIndices()[0] == iv && prior->getAttrs().empty()) {
            forwarded.emplace_back(load, prior.getResult());
            return WalkResult::advance();
          }
        }
      }
    } else if (isa<arith::ConstantOp, arith::CmpIOp, arith::ExtSIOp,
                   arith::ExtUIOp, arith::AndIOp>(op)) {
      pure.push_back(op);
      return WalkResult::advance();
    }
    return WalkResult(reject(op, "write region contains unsupported effects, reads or attributes"));
  });
  if (checked.wasInterrupted()) return failure();
  for (auto [load, replacement] : forwarded) {
    load.replaceAllUsesWith(replacement);
    load.erase();
  }
  for (Operation *op : pure) op->moveBefore(split);
  OpBuilder builder(split);
  auto loc = split.getLoc();
  auto negate = [&](Value cond) -> Value {
    if (auto cmp = cond.getDefiningOp<arith::CmpIOp>())
      return builder.create<arith::CmpIOp>(loc, arith::invertPredicate(cmp.getPredicate()),
                                          cmp.getLhs(), cmp.getRhs());
    auto inverse = builder.create<arith::XOrIOp>(loc, cond,
        builder.create<arith::ConstantIntOp>(loc, 1, 1));
    generated.insert(inverse);
    return inverse;
  };
  // A null suffix means that this arm always writes once entered.
  auto suffix = [&](memref::StoreOp store) -> Value {
    Value path;
    Operation *child = store;
    for (auto parent = child->getParentOp(); parent != split;
         child = parent, parent = parent->getParentOp()) {
      auto branch = cast<scf::IfOp>(parent);
      Value term = branch.getCondition();
      if (child->getParentRegion() == &branch.getElseRegion()) term = negate(term);
      path = path ? builder.create<arith::AndIOp>(loc, path, term).getResult() : term;
    }
    return path;
  };
  Value yes = suffix(left), no = suffix(right);
  Value gate = split.getCondition();
  // Prefer a single shared predicate for both value and enable. This also
  // canonicalizes a flipped fork when exactly one arm is unconditional.
  if (yes && !no) {
    std::swap(yes, no);
    std::swap(left, right);
    gate = negate(gate);
  }
  selectedData = builder.create<arith::SelectOp>(loc, gate, left.getValue(), right.getValue());
  Value enabled;
  if (yes && no) {
    Value takeLeft = builder.create<arith::AndIOp>(loc, gate, yes);
    Value takeRight = builder.create<arith::AndIOp>(loc, negate(gate), no);
    enabled = builder.create<arith::OrIOp>(loc, takeLeft, takeRight);
  } else if (no) {
    enabled = builder.create<arith::OrIOp>(loc, gate, no);
  }
  generated.insert(selectedData.getDefiningOp());
  if (enabled) {
    generated.insert(enabled.getDefiningOp());
    auto write = builder.create<scf::IfOp>(loc, enabled, false);
    builder.setInsertionPointToStart(&write.getThenRegion().front());
  }
  builder.create<memref::StoreOp>(loc, selectedData, left.getMemref(), left.getIndices());
  split.erase();
  return success();
}
