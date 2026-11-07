// Host-wide proof is deliberately separate from device address narrowing.
// No external descriptor is accepted as proof: this routine derives every
// address/effect fact from typed SSA and then constructs a fresh local function.
LogicalResult LowerCEntryPass::lowerLarge(func::FuncOp function,
                                         SmallVector<Object> &objects,
                                         llvm::json::Object &root) {
  auto reject = [&](Twine message) -> LogicalResult {
    function.emitError("large C-entry: ") << message; return failure();
  };
  for (auto attr : function->getAttrs()) {
    auto name = attr.getName().strref();
    if (name != "sym_name" && name != "function_type" && name != "llvm.linkage")
      return reject("unsupported function or parameter attributes");
  }
  if (!getOperation()->getAttrs().empty())
    return reject("unsupported module attributes; target metadata must be retained separately");
  auto plan = root.getObject("host_plan");
  if (!plan || plan->size() != 4 || plan->getString("strategy") != "per-query" ||
      plan->getInteger("tile_items") != 256 || plan->getString("ownership") != "copy")
    return reject("requires per-query, 256-item, owned-copy host_plan");
  auto binds = plan->getArray("bind_arguments");
  if (!binds || binds->size() != 3 || objects.size() != 5)
    return reject("requires three bound inputs and five objects");
  SmallVector<unsigned> bound;
  for (auto &v : *binds) {
    auto pos = v.getAsInteger();
    if (!pos || *pos < 0 || *pos >= 5 || llvm::is_contained(bound, unsigned(*pos)))
      return reject("bound argument positions must be distinct");
    bound.push_back(*pos);
  }
  // Expand affine address maps without relying on textual SSA names.
  SmallVector<Operation *> affineMemory;
  function.walk([&](Operation *op) {
    if (isa<affine::AffineLoadOp, affine::AffineStoreOp>(op)) affineMemory.push_back(op);
  });
  for (Operation *op : affineMemory) {
    for (auto attr : op->getAttrs())
      if (attr.getName().strref() != "map") return reject("unsupported memory attributes");
    OpBuilder b(op); AffineMap map; ValueRange operands;
    if (auto l = dyn_cast<affine::AffineLoadOp>(op)) { map=l.getAffineMap(); operands=l.getMapOperands(); }
    else { auto s=cast<affine::AffineStoreOp>(op); map=s.getAffineMap(); operands=s.getMapOperands(); }
    if (map.getNumResults()!=1) return reject("rank-one accesses required");
    std::function<Value(AffineExpr)> expand = [&](AffineExpr e)->Value {
      if (auto c=dyn_cast<AffineConstantExpr>(e)) return b.create<arith::ConstantIndexOp>(op->getLoc(),c.getValue());
      if (auto d=dyn_cast<AffineDimExpr>(e)) return operands[d.getPosition()];
      if (auto s=dyn_cast<AffineSymbolExpr>(e)) return operands[map.getNumDims()+s.getPosition()];
      auto x=dyn_cast<AffineBinaryOpExpr>(e);
      if (!x || (e.getKind()!=AffineExprKind::Add && e.getKind()!=AffineExprKind::Mul)) return {};
      Value l=expand(x.getLHS()),r=expand(x.getRHS()); if (!l || !r) return {};
      if (e.getKind()==AffineExprKind::Add) return b.create<arith::AddIOp>(op->getLoc(),l,r);
      return b.create<arith::MulIOp>(op->getLoc(),l,r);
    };
    Value index=expand(map.getResult(0)); if (!index) return reject("unsupported address map");
    if (auto l=dyn_cast<affine::AffineLoadOp>(op)) {
      auto n=b.create<memref::LoadOp>(op->getLoc(),l.getMemRef(),ValueRange{index}); l.replaceAllUsesWith(n.getResult());
    } else { auto s=cast<affine::AffineStoreOp>(op); b.create<memref::StoreOp>(op->getLoc(),s.getValue(),s.getMemRef(),ValueRange{index}); }
    op->erase();
  }
  Operation *loop=nullptr; Value iv; int64_t trips=1;
  for (Operation &op : function.front()) {
    if (auto l=dyn_cast<scf::ForOp>(op)) {
      if (loop || constant(l.getLowerBound())!=0 || constant(l.getStep())!=1 ||
          !constant(l.getUpperBound()) || l.getNumResults()) return reject("requires one fixed zero-based unit-step loop");
      loop=l; iv=l.getInductionVar(); trips=*constant(l.getUpperBound());
    } else if (auto l=dyn_cast<affine::AffineForOp>(op)) {
      if (loop || !l.hasConstantBounds() || l.getConstantLowerBound()!=0 || l.getStep()!=1 || l.getNumResults())
        return reject("requires one fixed zero-based unit-step loop");
      loop=l;iv=l.getInductionVar();trips=l.getConstantUpperBound();
    }
  }
  if (trips<1 || trips>16777216) return reject("trip count outside supported 1..16777216 resource bound");
  auto isIteration = [&](Value index) { return loop ? index==iv : constant(index)==0; };
  DenseMap<Value,unsigned> positions;
  for (auto a:function.getArguments()) positions[a]=a.getArgNumber();
  SmallVector<memref::LoadOp> loads; SmallVector<memref::StoreOp> stores;
  SmallVector<scf::IfOp> branches; bool bad=false;
  function.walk([&](Operation *op) {
    if (op==function.getOperation()) return;
    if (!isa<arith::ConstantOp,arith::ExtSIOp,arith::ExtUIOp,arith::IndexCastOp,
             arith::AddIOp,arith::MulIOp,arith::CmpIOp,arith::AndIOp,arith::SelectOp,arith::TruncIOp,
             memref::LoadOp,memref::StoreOp,scf::IfOp,scf::YieldOp,
             affine::AffineYieldOp,func::ReturnOp>(op) && op!=loop) { bad=true; return; }
    if (loop && op!=loop && !loop->isAncestor(op) && !isa<arith::ConstantOp,func::ReturnOp>(op)) bad=true;
    for (auto attr:op->getAttrs()) {
      auto name=attr.getName().strref();
      if ((isa<arith::ConstantOp>(op) && name=="value") ||
          (isa<arith::CmpIOp>(op) && name=="predicate") ||
          (isa<arith::AddIOp,arith::MulIOp>(op) && name=="overflowFlags") ||
          (isa<affine::AffineForOp>(op) && (name=="lowerBoundMap" || name=="upperBoundMap" || name=="step" || name=="operandSegmentSizes"))) continue;
      if (isa<memref::LoadOp,memref::StoreOp>(op) && name=="nontemporal") {
        auto flag=dyn_cast<BoolAttr>(attr.getValue()); if (flag && !flag.getValue()) continue;
      }
      bad=true;
    }
    if (auto l=dyn_cast<memref::LoadOp>(op)) loads.push_back(l);
    if (auto s=dyn_cast<memref::StoreOp>(op)) stores.push_back(s);
    if (auto branch=dyn_cast<scf::IfOp>(op)) {
      if (branch.getNumResults()>1 || (!branch.getNumResults() && !branch.getElseRegion().empty())) bad=true;
      branches.push_back(branch);
    }
    if (isa<affine::AffineYieldOp,func::ReturnOp>(op) && op->getNumOperands()) bad=true;
  });
  if (bad || loads.size()!=4 || stores.size()!=1) return reject("unsupported operations, attributes, or effects; requires four reads and one store");
  auto store=stores.front();
  if (!positions.count(store.getMemref()) || store.getIndices().size()!=1 || !isIteration(store.getIndices()[0]))
    return reject("unique destination must be the same-iteration contiguous store");
  unsigned dst=positions[store.getMemref()];
  memref::LoadOp sourceLoad;
  for (auto load:loads)
    if (positions.count(load.getMemref()) && !llvm::is_contained(bound,positions[load.getMemref()])) {
      if (sourceLoad) return reject("requires a unique source read");
      sourceLoad=load;
    }
  if (!sourceLoad || !positions.count(sourceLoad.getMemref()) || sourceLoad.getIndices().size()!=1)
    return reject("destination must receive the unique source loaded value or its proven range selection");
  unsigned src=positions[sourceLoad.getMemref()];
  if (src==dst || llvm::is_contained(bound,src) || llvm::is_contained(bound,dst))
    return reject("source, destination and bound arrays must be distinct objects");
  Value leaves[3];
  for (auto load:loads) {
    if (load==sourceLoad) continue;
    if (!positions.count(load.getMemref()) || load.getIndices().size()!=1 || !isIteration(load.getIndices()[0]))
      return reject("bound reads must use the same iteration index");
    auto found=llvm::find(bound,positions[load.getMemref()]);
    if (found==bound.end() || leaves[found-bound.begin()]) return reject("one read of each bound input required");
    leaves[found-bound.begin()]=load.getResult();
  }
  for (unsigned i=0;i<5;++i) {
    auto &o=objects[i];
    if (i==src) { if (o.write || o.dtype!="uint16") return reject("source must be read-only uint16"); }
    else if (i==dst) { if (!o.partial || o.dtype!="uint16" || o.elements!=trips) return reject("destination must be uint16 inout with exact trip count"); }
    else if (o.write || o.dtype!="int16" || o.elements!=trips) return reject("bound inputs must be signed int16 with exact trip count");
  }
  // Address polynomial uses i64/index operations only. Signed extensions keep
  // the actual coordinate interpretation; truncation and unsigned casts reject.
  struct Polynomial { int64_t x=0,y=0,c=0; };
  unsigned steps=0;
  std::function<std::optional<Polynomial>(Value)> polynomial = [&](Value v)->std::optional<Polynomial> {
    if (++steps>128) return std::nullopt;
    if (auto k=constant(v)) { if (*k<0 || *k>16777216) return std::nullopt; return Polynomial{0,0,*k}; }
    if (auto e=v.getDefiningOp<arith::ExtSIOp>()) {
      if (!v.getType().isInteger(64) || !e.getIn().getType().isInteger(16)) return std::nullopt;
      if(e.getIn()==leaves[0]) return Polynomial{1,0,0};
      if(e.getIn()==leaves[1]) return Polynomial{0,1,0};
      return std::nullopt;
    }
    if (auto c=v.getDefiningOp<arith::IndexCastOp>()) {
      if (!v.getType().isIndex()) return std::nullopt;
      // cgeist folds the one-trip affine case directly to signed index casts.
      if (c.getIn() == leaves[0]) return Polynomial{1,0,0};
      if (c.getIn() == leaves[1]) return Polynomial{0,1,0};
      if (!c.getIn().getType().isInteger(64)) return std::nullopt;
      return polynomial(c.getIn());
    }
    auto op=v.getDefiningOp();
    if (!op || !isa<arith::AddIOp,arith::MulIOp>(op) || (!v.getType().isInteger(64) && !v.getType().isIndex())) return std::nullopt;
    auto l=polynomial(op->getOperand(0)),r=polynomial(op->getOperand(1)); if(!l||!r) return std::nullopt;
    Polynomial n;
    if(isa<arith::AddIOp>(op)) n={l->x+r->x,l->y+r->y,l->c+r->c};
    else {
      if (l->x || l->y) std::swap(l,r);
      if (l->x || l->y) return std::nullopt;
      n={l->c*r->x,l->c*r->y,l->c*r->c};
    }
    if(n.x>16777216 || n.y>16777216 || n.c>16777216) return std::nullopt;
    return n;
  };
  auto address=polynomial(sourceLoad.getIndices()[0]);
  if (!address || address->x!=1 || address->y<1 || address->y>32767 || address->c)
    return reject("global address must be a proven wide y*width+x expression");
  int64_t width=address->y;
  if(objects[src].elements%width) return reject("source shape must be width*height");
  int64_t height=objects[src].elements/width;
  if(height<1 || height>32767) return reject("height outside signed-coordinate bounds");
  bool rangeMode=store.getValue()!=sourceLoad.getResult();
  int64_t rangeLo=0,rangeHi=0;
  llvm::SmallPtrSet<Operation *,16> valueOps;
  if(rangeMode) {
    // This finite-domain proof follows typed SSA, including sign extension,
    // truncation and branch yields. It never reinterprets signed IR using a
    // host dtype assertion. Only two comparisons of this read are admitted.
    unsigned nodes=0, comparisons=0, lowerBounds=0,upperBounds=0;
    auto pixelProjection = [&](Value v)->bool {
      if(v==sourceLoad.getResult()) return true;
      if(auto e=v.getDefiningOp<arith::ExtUIOp>())
        return e.getIn()==sourceLoad.getResult() && e.getType().isInteger(32);
      if(auto e=v.getDefiningOp<arith::ExtSIOp>())
        return e.getIn()==sourceLoad.getResult() && e.getType().isInteger(32);
      return false;
    };
    std::function<bool(Value)> tree = [&](Value v)->bool {
      if(++nodes>128) return false;
      if(v==sourceLoad.getResult()) return true;
      auto op=v.getDefiningOp();
      auto type=dyn_cast<IntegerType>(v.getType());
      if(!op || !type || (type.getWidth()!=1 && type.getWidth()!=16 && type.getWidth()!=32)) return false;
      if(valueOps.contains(op)) return true;
      valueOps.insert(op);
      if(isa<arith::ConstantOp>(op)) return constant(v).has_value();
      if(isa<arith::ExtSIOp,arith::ExtUIOp,arith::TruncIOp>(op)) return tree(op->getOperand(0));
      if(auto cmp=dyn_cast<arith::CmpIOp>(op)) {
        Value l=cmp.getLhs(),r=cmp.getRhs();auto p=cmp.getPredicate();
        if(constant(l)) {std::swap(l,r);p=reverse(p);}
        auto c=constant(r);
        if(!pixelProjection(l)||!c) return false;
        using P=arith::CmpIPredicate;
        if(p==P::sge||p==P::uge) {rangeLo=*c;++lowerBounds;}
        else if(p==P::sgt||p==P::ugt) {rangeLo=*c+1;++lowerBounds;}
        else if(p==P::sle||p==P::ule) {rangeHi=*c;++upperBounds;}
        else if(p==P::slt||p==P::ult) {rangeHi=*c-1;++upperBounds;}
        else return false;
        ++comparisons;
        return tree(l)&&tree(r);
      }
      if(auto a=dyn_cast<arith::AndIOp>(op))
        return a.getType().isInteger(1)&&tree(a.getLhs())&&tree(a.getRhs());
      if(auto s=dyn_cast<arith::SelectOp>(op))
        return tree(s.getCondition())&&tree(s.getTrueValue())&&tree(s.getFalseValue());
      if(auto branch=dyn_cast<scf::IfOp>(op)) {
        if(branch.getNumResults()!=1||branch.getElseRegion().empty()) return false;
        auto t=dyn_cast<scf::YieldOp>(branch.getThenRegion().front().getTerminator());
        auto f=dyn_cast<scf::YieldOp>(branch.getElseRegion().front().getTerminator());
        return t&&f&&t.getNumOperands()==1&&f.getNumOperands()==1&&
          tree(branch.getCondition())&&tree(t.getOperand(0))&&tree(f.getOperand(0));
      }
      return false;
    };
    if(!tree(store.getValue())||comparisons!=2||lowerBounds!=1||upperBounds!=1||
       rangeLo<1||rangeLo>rangeHi||rangeHi>65535)
      return reject("range effect requires two constant bounds on the same uint16 read and a pure value merge");
    std::function<llvm::APInt(Value,unsigned)> evaluate = [&](Value v,unsigned pixel)->llvm::APInt {
      if(v==sourceLoad.getResult()) return llvm::APInt(16,pixel);
      auto op=v.getDefiningOp();
      unsigned bits=cast<IntegerType>(v.getType()).getWidth();
      if(auto c=constant(v)) return llvm::APInt(bits,*c);
      if(auto e=dyn_cast<arith::ExtUIOp>(op)) return evaluate(e.getIn(),pixel).zext(bits);
      if(auto e=dyn_cast<arith::ExtSIOp>(op)) return evaluate(e.getIn(),pixel).sext(bits);
      if(auto e=dyn_cast<arith::TruncIOp>(op)) return evaluate(e.getIn(),pixel).trunc(bits);
      if(auto a=dyn_cast<arith::AndIOp>(op)) return evaluate(a.getLhs(),pixel)&evaluate(a.getRhs(),pixel);
      if(auto c=dyn_cast<arith::CmpIOp>(op)) {
        auto l=evaluate(c.getLhs(),pixel),r=evaluate(c.getRhs(),pixel);bool yes=false;
        using P=arith::CmpIPredicate;
        switch(c.getPredicate()) {
        case P::sge:yes=l.sge(r);break;case P::sgt:yes=l.sgt(r);break;
        case P::sle:yes=l.sle(r);break;case P::slt:yes=l.slt(r);break;
        case P::uge:yes=l.uge(r);break;case P::ugt:yes=l.ugt(r);break;
        case P::ule:yes=l.ule(r);break;case P::ult:yes=l.ult(r);break;
        default:llvm_unreachable("validated range comparison");
        }
        return llvm::APInt(1,yes);
      }
      if(auto s=dyn_cast<arith::SelectOp>(op))
        return evaluate(evaluate(s.getCondition(),pixel).isZero()?s.getFalseValue():s.getTrueValue(),pixel);
      auto branch=cast<scf::IfOp>(op);
      auto &region=evaluate(branch.getCondition(),pixel).isZero()?branch.getElseRegion():branch.getThenRegion();
      return evaluate(region.front().getTerminator()->getOperand(0),pixel);
    };
    for(unsigned v=0;v<65536;++v)
      if(evaluate(store.getValue(),v)!=llvm::APInt(16,rangeLo<=v&&v<=rangeHi?v:0))
        return reject("range expression is not uint16 inclusive-range value-or-zero for every input code");
  }
  // Exact boolean conjunctions, not merely range sufficiency: host skip and
  // device write conditions must be equivalent to the original control path.
  auto path = [&](Operation *op, SmallVector<Value> &conditions)->bool {
    for(Operation *child=op,*parent=op->getParentOp(); parent!=(loop ? loop : function.getOperation()); child=parent,parent=parent->getParentOp()) {
      auto branch=dyn_cast_or_null<scf::IfOp>(parent);
      if(!branch || child->getParentRegion()!=&branch.getThenRegion()) return false;
      conditions.push_back(branch.getCondition());
    }
    return true;
  };
  auto leaf = [&](Value v)->Value {
    if(auto e=v.getDefiningOp<arith::ExtSIOp>())
      if(e.getIn().getType().isInteger(16) && e.getType().isInteger(32)) return e.getIn();
    if(auto e=v.getDefiningOp<arith::ExtUIOp>())
      if(e.getIn()==sourceLoad.getResult() && e.getType().isInteger(32)) return e.getIn();
    return v;
  };
  auto classify = [&](ArrayRef<Value> conditions, bool allowPixel)->std::optional<unsigned> {
    unsigned mask=0,count=0;
    std::function<bool(Value)> visit = [&](Value v)->bool {
      if(++count>128) return false;
      if(auto a=v.getDefiningOp<arith::AndIOp>()) return a.getType().isInteger(1) && visit(a.getLhs()) && visit(a.getRhs());
      auto cmp=v.getDefiningOp<arith::CmpIOp>(); if(!cmp) return false;
      Value l=leaf(cmp.getLhs()),r=cmp.getRhs(); auto p=cmp.getPredicate();
      if(constant(l)) { r=cmp.getLhs();l=leaf(cmp.getRhs());p=reverse(p); }
      auto c=constant(r); if(!c) return false;
      using P=arith::CmpIPredicate;
      if(l==leaves[2] && p==P::ne && *c==0) {mask|=1;return true;}
      if(l==leaves[0] && ((p==P::sge && *c==0)||(p==P::sgt && *c==-1))) {mask|=2;return true;}
      if(l==leaves[1] && ((p==P::sge && *c==0)||(p==P::sgt && *c==-1))) {mask|=4;return true;}
      if(l==leaves[0] && ((p==P::slt && *c==width)||(p==P::sle && *c==width-1))) {mask|=8;return true;}
      if(l==leaves[1] && ((p==P::slt && *c==height)||(p==P::sle && *c==height-1))) {mask|=16;return true;}
      if(allowPixel && l==sourceLoad.getResult() && p==P::ne && *c==0) {mask|=32;return true;}
      return false;
    };
    for(Value c:conditions) if(!visit(c)) return std::nullopt;
    return mask;
  };
  SmallVector<Value> readPath,writePath;
  if(!path(sourceLoad,readPath) || !path(store,writePath)) return reject("access requires a nested true path");
  auto readMask=classify(readPath,false),writeMask=classify(writePath,true);
  if(!readMask || *readMask!=31 || !writeMask || (*writeMask!=31 && *writeMask!=63))
    return reject("read guard or write predicate does not match the proven address semantics");
  for(auto branch:branches)
    if(!branch->isAncestor(sourceLoad) && !branch->isAncestor(store) && !valueOps.contains(branch)) return reject("unrelated control effects unsupported");
  bool nonzero=*writeMask==63;
  if(rangeMode && nonzero) return reject("range selection must write for every valid address, including zero");
  std::string entry=function.getSymName().str();
  llvm::json::Object proof{{"schema","adora.guarded-inplace-large.v1"},{"entry",entry},
    {"width",width},{"height",height},{"items",trips},{"source_elements",objects[src].elements},
    {"tile_items",256},{"value_nonzero",nonzero},{"source_argument",int64_t(src)},
    {"value_mode",rangeMode?"range":nonzero?"nonzero":"direct"},
    {"range_lo",rangeMode?llvm::json::Value(rangeLo):llvm::json::Value(nullptr)},
    {"range_hi",rangeMode?llvm::json::Value(rangeHi):llvm::json::Value(nullptr)},
    {"destination_argument",int64_t(dst)},{"x_argument",int64_t(bound[0])},
    {"y_argument",int64_t(bound[1])},{"enable_argument",int64_t(bound[2])},
    {"global_address_bits",64},{"device_element_bits",16},{"proof","typed SSA exact guards, wide address and unique value effect; range exhaustively checked over uint16"}};
  // Construct a local semantic equivalent only after the global proof passes.
  auto loc=function.getLoc(); OpBuilder b(function.getContext());
  function.getBody().getBlocks().clear();
  auto type=MemRefType::get({256},b.getI16Type());
  function.setType(b.getFunctionType({type,type,type,type},{}));
  function.setSymName("_adora_large_tile");
  auto *body=function.addEntryBlock(); b.setInsertionPointToStart(body);
  auto zero=b.create<arith::ConstantIntOp>(loc,0,16);
  auto limit=b.create<arith::ConstantIntOp>(loc,256,16);
  auto localLoop=b.create<affine::AffineForOp>(loc,0,256);
  b.setInsertionPoint(localLoop.getBody()->getTerminator());
  auto idx=b.create<memref::LoadOp>(loc,function.getArgument(1),ValueRange{localLoop.getInductionVar()});
  auto en=b.create<memref::LoadOp>(loc,function.getArgument(2),ValueRange{localLoop.getInductionVar()});
  auto active=b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::ne,en,zero);
  auto lo=b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::sge,idx,zero);
  auto hi=b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::slt,idx,limit);
  auto bounds=b.create<arith::AndIOp>(loc,lo,hi);
  auto guard=b.create<arith::AndIOp>(loc,active,bounds);
  auto branch=b.create<scf::IfOp>(loc,guard,false); b.setInsertionPointToStart(&branch.getThenRegion().front());
  auto addressIndex=b.create<arith::IndexCastOp>(loc,b.getIndexType(),idx);
  auto pixel=b.create<memref::LoadOp>(loc,function.getArgument(0),ValueRange{addressIndex});
  if(nonzero) {
    auto nz=b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::ne,pixel,zero);
    auto inner=b.create<scf::IfOp>(loc,nz,false); b.setInsertionPointToStart(&inner.getThenRegion().front());
  }
  Value data=pixel;
  if(rangeMode) {
    // unsigned16 v is ordered exactly by signed16((v + 0x8000) mod 65536).
    // This add intentionally carries no nsw/nuw flags.
    auto bias=b.create<arith::ConstantIntOp>(loc,-32768,16);
    auto biased=b.create<arith::AddIOp>(loc,pixel,bias);
    auto lower=b.create<arith::ConstantIntOp>(loc,rangeLo-32768,16);
    auto upper=b.create<arith::ConstantIntOp>(loc,rangeHi-32768,16);
    auto above=b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::sge,biased,lower);
    auto below=b.create<arith::CmpIOp>(loc,arith::CmpIPredicate::sle,biased,upper);
    auto inside=b.create<arith::AndIOp>(loc,above,below);
    auto selection=b.create<scf::IfOp>(loc,TypeRange{b.getI16Type()},inside,true);
    b.setInsertionPointToStart(&selection.getThenRegion().front());
    b.create<scf::YieldOp>(loc,ValueRange{pixel});
    b.setInsertionPointToStart(&selection.getElseRegion().front());
    b.create<scf::YieldOp>(loc,ValueRange{zero});
    b.setInsertionPointAfter(selection);
    data=selection.getResult(0);
  }
  b.create<memref::StoreOp>(loc,data,function.getArgument(3),ValueRange{localLoop.getInductionVar()});
  b.setInsertionPointToEnd(body); b.create<func::ReturnOp>(loc);
  SmallVector<Object> localObjects{{256,"uint16",false},{256,"int16",false},{256,"int16",false},{256,"uint16",true,true}};
  if(failed(lower(function,localObjects,rangeMode))) return failure();
  std::error_code ec; llvm::raw_fd_ostream out(descriptor,ec,llvm::sys::fs::OF_Text);
  if(ec) return reject("cannot write verified descriptor");
  out << llvm::formatv("{0:2}\n",llvm::json::Value(std::move(proof)));
  return success();
}
