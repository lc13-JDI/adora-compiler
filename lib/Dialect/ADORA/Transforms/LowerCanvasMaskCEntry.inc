// The private object is proved before it is removed. The projection below is
// internal compiler state, never a contract assertion or accepted user marker.
LogicalResult LowerCEntryPass::lowerCanvas(func::FuncOp function,
    SmallVector<Object> &objects, llvm::json::Object &root) {
  auto reject = [&](Twine message) -> LogicalResult {
    function.emitError("canvas C-entry: ") << message; return failure();
  };
  if (objects.size() != 6) return reject("requires six external objects");
  SmallVector<scf::ForOp> loops;
  memref::AllocaOp allocation;
  for (Operation &op : function.front()) {
    if (auto loop = dyn_cast<scf::ForOp>(op)) loops.push_back(loop);
    else if (auto local = dyn_cast<memref::AllocaOp>(op)) {
      if (allocation) return reject("requires one private allocation");
      allocation = local;
    } else if (!isa<arith::ConstantOp, func::ReturnOp>(op))
      return reject("effects outside the two loops are unsupported");
  }
  if (loops.size() != 2 || !allocation)
    return reject("requires two ordered loops and one private canvas");
  auto first = loops[0], second = loops[1];
  auto trips = constant(first.getUpperBound());
  for (auto loop : loops)
    if (!trips || *trips <= 0 || *trips > 16777216 ||
        constant(loop.getLowerBound()) != 0 || constant(loop.getStep()) != 1 ||
        constant(loop.getUpperBound()) != trips || loop.getNumResults() || !loop->getAttrs().empty())
      return reject("canvas loops require identical positive static domains and unit steps");
  auto type = allocation.getType();
  if (!allocation->isBeforeInBlock(first) || type.getRank() != 1 ||
      type.getDimSize(0) != *trips || !type.getElementType().isInteger(16) ||
      !type.getLayout().isIdentity() || type.getMemorySpace() ||
      allocation.getDynamicSizes().size() || allocation.getSymbolOperands().size())
    return reject("private canvas must be a fixed whole i16 array covering the loop domain");
  for (auto attr : allocation->getAttrs())
    if (attr.getName().strref() != "alignment" && attr.getName().strref() != "operandSegmentSizes")
      return reject("unsupported private allocation attributes");
  SmallVector<memref::StoreOp> writes;
  memref::LoadOp consumer;
  unsigned uses = 0;
  for (OpOperand &use : allocation.getResult().getUses()) {
    ++uses;
    auto *op = use.getOwner();
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      if (store.getMemref() != allocation.getResult() || !first->isAncestor(op) ||
          store.getIndices().size() != 1 || store.getIndices()[0] != first.getInductionVar())
        return reject("private canvas writes must use the producer iteration index");
      writes.push_back(store);
    } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
      if (consumer || load.getMemref() != allocation.getResult() || op->getParentOp() != second ||
          load.getIndices().size() != 1 || load.getIndices()[0] != second.getInductionVar())
        return reject("canvas must have exactly one same-slot consumer in the second loop");
      consumer = load;
    } else return reject("private canvas escapes or has a view, copy, call, or unknown user");
    for (auto attr : op->getAttrs()) {
      auto flag = dyn_cast<BoolAttr>(attr.getValue());
      if (attr.getName().strref() != "nontemporal" || !flag || flag.getValue())
        return reject("unsupported canvas memory effects or attributes");
    }
  }
  if (!consumer || (writes.size() != 1 && writes.size() != 2))
    return reject("canvas requires complete initialization and one optional update");
  memref::StoreOp output;
  for (Operation &op : *second.getBody()) if (auto store = dyn_cast<memref::StoreOp>(op)) {
    if (output) return reject("mask has an extra consumer or write");
    output = store;
  }
  auto destination = output ? dyn_cast<BlockArgument>(output.getMemref()) : BlockArgument();
  if (!destination || destination.getOwner() != &function.front() ||
      output.getIndices().size() != 1 || output.getIndices()[0] != second.getInductionVar())
    return reject("mask must be a complete same-slot external output");
  unsigned dst = destination.getArgNumber();
  if (!objects[dst].write || objects[dst].partial || objects[dst].dtype != "uint16" || objects[dst].elements != *trips)
    return reject("mask must be write-only uint16 with exact trip count");
  // Exact typed finite-domain proof of the only consumer. No effect-bearing
  // operation or unexamined value can hide in the second loop.
  DenseSet<Operation *> valueOps;
  unsigned nodes = 0;
  std::function<bool(Value)> tree = [&](Value value) {
    if (++nodes > 128) return false;
    if (value == consumer.getResult()) return true;
    auto *op = value.getDefiningOp();
    auto integer = dyn_cast<IntegerType>(value.getType());
    if (!op || !integer || (integer.getWidth() != 1 && integer.getWidth() != 16 && integer.getWidth() != 32)) return false;
    if (valueOps.contains(op)) return true;
    valueOps.insert(op);
    if (isa<arith::ConstantOp>(op)) return constant(value).has_value();
    if (!isa<arith::ExtUIOp,arith::ExtSIOp,arith::TruncIOp,arith::CmpIOp,arith::SelectOp>(op)) return false;
    if (auto cmp = dyn_cast<arith::CmpIOp>(op))
      if (cmp.getPredicate() != arith::CmpIPredicate::eq && cmp.getPredicate() != arith::CmpIPredicate::ne &&
          cmp.getPredicate() != arith::CmpIPredicate::sgt) return false;
    return llvm::all_of(op->getOperands(), [&](Value operand) { return tree(operand); });
  };
  if (!tree(output.getValue())) return reject("mask consumer is not a supported typed nonzero selection");
  for (Operation &op : *second.getBody()) {
    if (&op != consumer && &op != output && !isa<scf::YieldOp>(op) && !valueOps.contains(&op))
      return reject("second loop has an extra consumer or effect");
    for (auto attr : op.getAttrs())
      if (!((isa<arith::ConstantOp>(op) && attr.getName().strref() == "value") ||
            (isa<arith::CmpIOp>(op) && attr.getName().strref() == "predicate")))
        return reject("unsupported mask consumer attributes");
  }
  std::function<llvm::APInt(Value,unsigned)> evaluate = [&](Value value,unsigned code) -> llvm::APInt {
    if (value == consumer.getResult()) return llvm::APInt(16,code);
    unsigned bits = cast<IntegerType>(value.getType()).getWidth();
    if (auto c = constant(value)) return llvm::APInt(bits,*c);
    auto *op = value.getDefiningOp();
    if (auto ext = dyn_cast<arith::ExtUIOp>(op)) return evaluate(ext.getIn(),code).zext(bits);
    if (auto ext = dyn_cast<arith::ExtSIOp>(op)) return evaluate(ext.getIn(),code).sext(bits);
    if (auto trunc = dyn_cast<arith::TruncIOp>(op)) return evaluate(trunc.getIn(),code).trunc(bits);
    if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
      auto lhs = evaluate(cmp.getLhs(),code);
      auto rhs = evaluate(cmp.getRhs(),code);
      if (cmp.getPredicate() == arith::CmpIPredicate::sgt)
        return llvm::APInt(1,lhs.sgt(rhs));
      return llvm::APInt(1,cmp.getPredicate() == arith::CmpIPredicate::eq ? lhs == rhs : lhs != rhs);
    }
    auto select = cast<arith::SelectOp>(op);
    return evaluate(evaluate(select.getCondition(),code).isZero() ? select.getFalseValue() : select.getTrueValue(),code);
  };
  for (unsigned code = 0; code < 65536; ++code)
    if (evaluate(output.getValue(),code) != llvm::APInt(16,code ? 255 : 0))
      return reject("mask must equal uint16(nonzero ? 255 : 0) for every canvas code");
  memref::LoadOp old;
  memref::StoreOp initialization, update;
  scf::IfOp merge;
  if (writes.size() == 2) {
    for (auto write : writes)
      if (write->getParentOp() == first) initialization = write;
      else update = write;
    if (!initialization || !update)
      return reject("old initialization must precede the guarded update");
    Operation *outer = update;
    while (outer->getParentOp() != first) outer = outer->getParentOp();
    if (!initialization->isBeforeInBlock(outer))
      return reject("old initialization must precede the guarded update");
    old = initialization.getValue().getDefiningOp<memref::LoadOp>();
  } else {
    update = writes.front();
    merge = update.getValue().getDefiningOp<scf::IfOp>();
    if (update->getParentOp() != first || !merge || merge->getParentOp() != first ||
        merge.getNumResults() != 1 || !merge.getResult(0).getType().isInteger(16) ||
        merge.getElseRegion().empty() || !llvm::hasSingleElement(merge.getElseRegion().front()))
      return reject("single canvas write requires a pure guarded SSA merge with old fallback");
    auto yield = dyn_cast<scf::YieldOp>(merge.getElseRegion().front().getTerminator());
    if (!yield || yield.getNumOperands() != 1) return reject("missing old fallback");
    old = yield.getOperand(0).getDefiningOp<memref::LoadOp>();
  }
  auto oldArg = old ? dyn_cast<BlockArgument>(old.getMemref()) : BlockArgument();
  if (!oldArg || oldArg.getOwner() != &function.front() || old->getParentOp() != first ||
      old.getIndices().size() != 1 || old.getIndices()[0] != first.getInductionVar() ||
      !old.getResult().hasOneUse()) return reject("old must initialize every slot through one same-index read");
  for (auto attr : old->getAttrs()) {
    auto flag = dyn_cast<BoolAttr>(attr.getValue());
    if (attr.getName().strref() != "nontemporal" || !flag || flag.getValue())
      return reject("unsupported old memory attributes");
  }
  if (merge && (!merge->getAttrs().empty() ||
      !merge.getThenRegion().front().getTerminator()->getAttrs().empty() ||
      !merge.getElseRegion().front().getTerminator()->getAttrs().empty()))
    return reject("unsupported SSA merge attributes");
  unsigned oldPosition = oldArg.getArgNumber();
  if (oldPosition == dst || objects[oldPosition].write || objects[oldPosition].dtype != "uint16" ||
      objects[oldPosition].elements != *trips || !oldArg.hasOneUse())
    return reject("old must be a distinct read-only whole uint16 object");
  // Validate every remaining operation before deleting any producer structure.
  // lowerLarge repeats its exhaustive whitelist and exact effect census on the
  // projection, including the guard and range graph retained below.
  if (merge && !merge.getResult(0).hasOneUse()) return reject("SSA canvas value has an extra observer");
  CanvasProof evidence{oldPosition, {}, uses};
  for (unsigned i = 0; i < objects.size(); ++i) if (i != oldPosition) evidence.originalPositions.push_back(i);
  second.erase();
  if (merge) {
    OpBuilder b(merge);
    auto guarded = b.create<scf::IfOp>(merge.getLoc(),merge.getCondition(),false);
    auto *yield = merge.getThenRegion().front().getTerminator();
    Value selected = yield->getOperand(0);
    auto &ops = merge.getThenRegion().front().getOperations();
    guarded.getThenRegion().front().getOperations().splice(guarded.getThenRegion().front().begin(),ops,ops.begin(),std::prev(ops.end()));
    b.setInsertionPoint(guarded.getThenRegion().front().getTerminator());
    b.create<memref::StoreOp>(update.getLoc(),selected,destination,ValueRange{first.getInductionVar()});
    update.erase(); merge.erase();
  } else {
    update.getMemrefMutable().assign(destination);
    initialization.erase();
  }
  old.erase(); allocation.erase();
  function.eraseArgument(oldPosition);
  objects.erase(objects.begin() + oldPosition);
  unsigned projectedDst = dst - (dst > oldPosition);
  objects[projectedDst].partial = true;
  auto plan = root.getObject("host_plan");
  if (!plan) return reject("missing host plan");
  auto binds = plan->getArray("bind_arguments");
  if (!binds) return reject("missing bound argument positions");
  for (auto &v : *binds) {
    auto pos = v.getAsInteger();
    if (!pos || *pos < 0 || *pos >= 6 || *pos == oldPosition || *pos == dst)
      return reject("bound arguments cannot name old or mask");
    v = *pos - (*pos > oldPosition);
  }
  return lowerLarge(function,objects,root,&evidence);
}

LogicalResult LowerCEntryPass::emitCanvas(func::FuncOp function, int64_t rangeLo, int64_t rangeHi) {
  auto loc = function.getLoc();
  OpBuilder b(function.getContext());
  auto type = MemRefType::get({256},b.getI16Type());
  auto build = [&](bool regular) -> LogicalResult {
    b.setInsertionPoint(function);
    const bool stream = regular && (canvasVariant == "C" || canvasVariant == "D");
    const bool predicate = regular && (canvasVariant == "B" || canvasVariant == "D");
    const unsigned arity = regular ? (stream ? 4 : 5) : 2;
    SmallVector<Type> types(arity,type);
    auto tile = b.create<func::FuncOp>(loc,regular ? "_adora_canvas_regular_tile" : "_adora_canvas_old_tile",b.getFunctionType(types,{}));
    auto *body = tile.addEntryBlock(); b.setInsertionPointToStart(body);
    auto zero = b.create<arith::ConstantIntOp>(loc,0,16);
    auto white = b.create<arith::ConstantIntOp>(loc,255,16);
    auto loop = b.create<affine::AffineForOp>(loc,0,256);
    b.setInsertionPoint(loop.getBody()->getTerminator());
    auto iv = loop.getInductionVar();
    auto old = b.create<memref::LoadOp>(loc,tile.getArgument(arity-2),ValueRange{iv});
    auto select = [&](Value cond, Value yes, Value no) -> Value {
      auto op = b.create<scf::IfOp>(loc,TypeRange{b.getI16Type()},cond,true);
      b.setInsertionPointToStart(&op.getThenRegion().front()); b.create<scf::YieldOp>(loc,ValueRange{yes});
      b.setInsertionPointToStart(&op.getElseRegion().front()); b.create<scf::YieldOp>(loc,ValueRange{no});
      b.setInsertionPointAfter(op);
      return op.getResult(0);
    };
    auto nonzeroMask = [&](Value value) -> Value {
      auto negative = b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::slt,value,zero);
      auto positive = b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::slt,zero,value);
      auto nonzero = b.create<arith::OrIOp>(loc,negative,positive);
      // The false operand is known to be zero. Use the routed input so
      // the hardware SELECT only needs one immediate.
      return select(nonzero,white,value);
    };
    Value result = old;
    if (regular) {
      Value idx, pixel, guard;
      if (!stream) idx = b.create<memref::LoadOp>(loc,tile.getArgument(1),ValueRange{iv});
      auto en = b.create<memref::LoadOp>(loc,tile.getArgument(arity-3),ValueRange{iv});
      if (stream) {
        // Private enabled is normalized to 0/1 after the original signed
        // geometry and nonzero-enable checks. Every packed slot is initialized.
        guard = b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::slt,zero,en);
        pixel = b.create<memref::LoadOp>(loc,tile.getArgument(0),ValueRange{iv});
      } else {
        auto limit = b.create<arith::ConstantIntOp>(loc,256,16);
        auto active = b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::ne,en,zero);
        auto lo = b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::sge,idx,zero);
        auto hi = b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::slt,idx,limit);
        auto bound = b.create<arith::AndIOp>(loc,lo,hi);
        guard = b.create<arith::AndIOp>(loc,active,bound);
      }
      // A false direct-mask predicate implies zero when enabled and old=0
      // otherwise. This routed fallback avoids two immediate SELECT operands.
      Value fallback;
      if (predicate) fallback = select(guard,zero,nonzeroMask(old));
      scf::IfOp branch;
      if (!stream) {
        branch = b.create<scf::IfOp>(loc,TypeRange{b.getI16Type()},guard,true);
        b.setInsertionPointToStart(&branch.getThenRegion().front());
        auto address = b.create<arith::IndexCastOp>(loc,b.getIndexType(),idx);
        pixel = b.create<memref::LoadOp>(loc,tile.getArgument(0),ValueRange{address});
      }
      auto bias = b.create<arith::ConstantIntOp>(loc,-32768,16);
      auto biased = b.create<arith::AddIOp>(loc,pixel,bias);
      auto low = b.create<arith::ConstantIntOp>(loc,rangeLo-32768,16);
      auto high = b.create<arith::ConstantIntOp>(loc,rangeHi-32768,16);
      auto above = b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::sge,biased,low);
      auto below = b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::sle,biased,high);
      auto inside = b.create<arith::AndIOp>(loc,above,below);
      if (stream && predicate) {
        auto hit = b.create<arith::AndIOp>(loc,guard,inside);
        result = select(hit,white,fallback);
      } else {
        result = predicate ? select(inside,white,fallback) : select(inside,pixel,zero);
        Value otherwise = predicate ? fallback : Value(old);
        if (stream) result = select(guard,result,otherwise);
        else {
          b.create<scf::YieldOp>(loc,ValueRange{result});
          b.setInsertionPointToStart(&branch.getElseRegion().front()); b.create<scf::YieldOp>(loc,ValueRange{otherwise});
          b.setInsertionPointAfter(branch);
          result = branch.getResult(0);
        }
      }
    }
    if (!predicate) result = nonzeroMask(result);
    b.create<memref::StoreOp>(loc,result,tile.getArgument(arity-1),ValueRange{iv});
    b.setInsertionPointToEnd(body); b.create<func::ReturnOp>(loc);
    SmallVector<Object> local;
    if (regular) local = {{256,"uint16",false},{256,"int16",false},{256,"int16",false},{256,"uint16",false},{256,"uint16",true}};
    else local = {{256,"uint16",false},{256,"uint16",true}};
    if (stream) local.erase(local.begin()+1);
    return lower(tile,local,true);
  };
  if (canvasKernel != "old-only" && failed(build(true))) return failure();
  if (canvasKernel != "regular" && failed(build(false))) return failure();
  return success();
}
