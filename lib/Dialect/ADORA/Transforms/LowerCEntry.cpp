// Structural bridge from a checked, static C pointer ABI to ADORA streams.
// This pass deliberately leaves guarded LOAD legalization to DFGgen.cpp.
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "PassDetail.h"
#include <algorithm>
#include <functional>
#include <optional>

using namespace mlir;
using namespace mlir::ADORA;

namespace {
struct Object {
  int64_t elements;
  std::string dtype;
  bool write;
  bool partial = false;
  unsigned reads = 0, writes = 0;
};
using Range = std::pair<int64_t, int64_t>;

static std::optional<int64_t> constant(Value value) {
  auto op = value.getDefiningOp<arith::ConstantOp>();
  auto attr = op ? dyn_cast<IntegerAttr>(op.getValue()) : IntegerAttr();
  if (!attr || attr.getValue().getBitWidth() > 64) return std::nullopt;
  return attr.getInt();
}

// Strip only the C signed integer promotion, never a general cast or i32 input.
static Value signedLeaf(Value value) {
  if (auto ext = value.getDefiningOp<arith::ExtSIOp>())
    if (ext.getIn().getType().isInteger(16) && ext.getType().isInteger(32))
      return ext.getIn();
  return value;
}

static arith::CmpIPredicate reverse( arith::CmpIPredicate p) {
  using P = arith::CmpIPredicate;
  switch (p) {
  case P::slt: return P::sgt;
  case P::sle: return P::sge;
  case P::sgt: return P::slt;
  case P::sge: return P::sle;
  default: return p;
  }
}

static void refine(Value guard, Value leaf, Range &range,
                   DenseSet<Value> &visited, unsigned depth = 0) {
  if (depth > 64 || !visited.insert(guard).second) return;
  if (auto both = guard.getDefiningOp<arith::AndIOp>()) {
    refine(both.getLhs(), leaf, range, visited, depth + 1);
    refine(both.getRhs(), leaf, range, visited, depth + 1);
    return;
  }
  auto cmp = guard.getDefiningOp<arith::CmpIOp>();
  if (!cmp) return;
  auto predicate = cmp.getPredicate();
  Value rhs = cmp.getRhs();
  if (signedLeaf(cmp.getLhs()) != leaf) {
    if (signedLeaf(rhs) != leaf) return;
    rhs = cmp.getLhs();
    predicate = reverse(predicate);
  }
  auto c = constant(rhs);
  if (!c) return;
  using P = arith::CmpIPredicate;
  switch (predicate) {
  case P::sge: range.first = std::max(range.first, *c); break;
  case P::sgt: range.first = std::max(range.first, *c + 1); break;
  case P::slt: range.second = std::min(range.second, *c - 1); break;
  case P::sle: range.second = std::min(range.second, *c); break;
  case P::eq: range.first = std::max(range.first, *c);
              range.second = std::min(range.second, *c); break;
  default: break;
  }
}

class LowerCEntryPass : public LowerCEntryBase<LowerCEntryPass> {
  void runOnOperation() override;
  LogicalResult lower(func::FuncOp function, SmallVector<Object> &objects);
  LogicalResult lowerLarge(func::FuncOp function, SmallVector<Object> &objects, llvm::json::Object &root);
};
} // namespace

void LowerCEntryPass::runOnOperation() {
  auto module = getOperation();
  auto reject = [&](Twine message) {
    module.emitError("C-entry: ") << message;
    signalPassFailure();
  };
  auto file = llvm::MemoryBuffer::getFile(contract);
  if (!file) return reject("cannot read object contract: " + contract);
  auto json = llvm::json::parse((*file)->getBuffer());
  if (!json) { llvm::consumeError(json.takeError()); return reject("invalid object contract JSON"); }
  auto root = json->getAsObject();
  if (!root) return reject("object contract must be an object");
  for (const auto &field : *root)
    if (field.first != "entry" && field.first != "defines" && field.first != "objects" &&
        field.first != "source" && field.first != "id" &&
        !(largeObjects && field.first == "host_plan"))
      return reject("unsupported object contract key: " + field.first.str());
  auto entry = root->getString("entry");
  auto list = root->getArray("objects");
  if (!entry || !list) return reject("object contract requires entry and objects");
  auto function = module.lookupSymbol<func::FuncOp>(*entry);
  if (!function || function.isExternal() || !function.getResultTypes().empty() ||
      !llvm::hasSingleElement(function.getBody()))
    return reject("entry must be one defined void function with a single block");
  if (function->hasAttr("adora.no_access") || function->hasAttr("adora.require_no_access") ||
      module->hasAttr("adora.no_access") || module->hasAttr("adora.require_no_access"))
    return reject("unsupported memory attributes (no-access/volatile)");
  if (list->size() != function.getNumArguments())
    return reject("object contract must describe every argument exactly once");
  if (bankBytes <= 0) return reject("bank-bytes must be positive");
  SmallVector<Object> objects(function.getNumArguments());
  SmallVector<bool> seen(function.getNumArguments(), false);
  for (const auto &item : *list) {
    auto object = item.getAsObject();
    if (!object) return reject("invalid object contract record");
    for (const auto &field : *object)
      if (field.first != "argument_position" && field.first != "elements" &&
          field.first != "host_dtype" && field.first != "access_assertion" &&
          field.first != "contiguous" && field.first != "name")
        return reject("unsupported object contract key: " + field.first.str());
    auto pos = object->getInteger("argument_position");
    auto elements = object->getInteger("elements");
    auto dtype = object->getString("host_dtype");
    auto access = object->getString("access_assertion");
    auto contiguous = object->getBoolean("contiguous");
    if (!pos || *pos < 0 || *pos >= objects.size() || seen[*pos])
      return reject("object contract argument positions must be unique and in range");
    if (!elements || *elements <= 0 || *elements > (largeObjects ? 16777216 : bankBytes / 2))
      return reject("object elements must be positive and fit one target bank");
    if (!dtype || (*dtype != "int16" && *dtype != "uint16") ||
        !access || (*access != "read" && *access != "write" && *access != "inout") ||
        !contiguous || !*contiguous)
      return reject("objects require int16/uint16 dtype, read/write/inout effects and contiguous=true");
    if (*access == "inout" && !allowPartialWrites)
      return reject("inout requires explicit allow-partial-writes=true");
    auto type = dyn_cast<MemRefType>(function.getArgument(*pos).getType());
    if (!type || type.getRank() != 1 || !type.getElementType().isSignlessInteger(16) ||
        !type.getLayout().isIdentity() || type.getMemorySpace())
      return reject("entry objects must be contiguous rank-one i16 memrefs");
    if (!type.isDynamicDim(0) && type.getDimSize(0) != *elements)
      return reject("object contract length contradicts static memref shape");
    seen[*pos] = true;
    objects[*pos] = {*elements, dtype->str(), *access != "read", *access == "inout"};
  }
  // Any other function could call the entry with the old ABI or introduce
  // effects outside the declared kernel. This explicit profile is one entry.
  for (auto other : module.getOps<func::FuncOp>())
    if (other != function) return reject("only the selected C entry is supported");
  if (largeObjects) {
    if (!allowPartialWrites || bankBytes < 512 || descriptor.empty())
      return reject("large objects require partial writes, a 512-byte bank and descriptor output");
    if (failed(lowerLarge(function, objects, *root))) signalPassFailure();
  } else if (failed(lower(function, objects))) signalPassFailure();
}

LogicalResult LowerCEntryPass::lower(func::FuncOp function,
                                   SmallVector<Object> &objects) {
  auto reject = [&](Operation *op, Twine message) -> LogicalResult {
    op->emitError("C-entry: ") << message;
    return failure();
  };
  bool unsupportedAttributes = false;
  function.walk([&](Operation *op) {
    if (op->getName().getDialectNamespace() != "arith") return;
    for (auto attr : op->getAttrs()) {
      StringRef name = attr.getName().strref();
      if ((isa<arith::ConstantOp>(op) && name == "value") ||
          (isa<arith::CmpIOp>(op) && name == "predicate") ||
          (isa<arith::AddIOp, arith::MulIOp>(op) && name == "overflowFlags")) continue;
      (void)reject(op, "unsupported arithmetic attributes");
      unsupportedAttributes = true;
    }
  });
  if (unsupportedAttributes) return failure();
  // cgeist may retain affine symbolic addresses or fold a one-iteration loop.
  // Expand only affine add/mul/constant maps into the same typed SSA slice
  // checked below. There is no interpretation of argument names or roles.
  SmallVector<Operation *> affineMemory;
  function.walk([&](Operation *op) {
    if (isa<affine::AffineLoadOp, affine::AffineStoreOp>(op)) affineMemory.push_back(op);
  });
  for (Operation *op : affineMemory) {
    for (auto attr : op->getAttrs())
      if (attr.getName().strref() != "map")
        return reject(op, "unsupported affine memory attributes");
    OpBuilder builder(op);
    AffineMap map;
    ValueRange operands;
    if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
      map = load.getAffineMap(); operands = load.getMapOperands();
    } else {
      auto store = cast<affine::AffineStoreOp>(op);
      map = store.getAffineMap(); operands = store.getMapOperands();
    }
    if (map.getNumResults() != 1) return reject(op, "requires rank-one affine memory maps");
    std::function<Value(AffineExpr)> expand = [&](AffineExpr expr) -> Value {
      if (auto c = dyn_cast<AffineConstantExpr>(expr))
        return builder.create<arith::ConstantIndexOp>(op->getLoc(), c.getValue());
      if (auto dim = dyn_cast<AffineDimExpr>(expr)) return operands[dim.getPosition()];
      if (auto sym = dyn_cast<AffineSymbolExpr>(expr)) return operands[map.getNumDims() + sym.getPosition()];
      auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
      if (!binary || (expr.getKind() != AffineExprKind::Add && expr.getKind() != AffineExprKind::Mul)) return {};
      Value lhs = expand(binary.getLHS()), rhs = expand(binary.getRHS());
      if (!lhs || !rhs) return {};
      if (expr.getKind() == AffineExprKind::Add)
        return builder.create<arith::AddIOp>(op->getLoc(), lhs, rhs);
      return builder.create<arith::MulIOp>(op->getLoc(), lhs, rhs);
    };
    Value index = expand(map.getResult(0));
    if (!index) return reject(op, "unsupported affine address expression; only add/mul are supported");
    if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
      auto replacement = builder.create<memref::LoadOp>(op->getLoc(), load.getMemRef(), ValueRange{index});
      load.replaceAllUsesWith(replacement.getResult());
    } else {
      auto store = cast<affine::AffineStoreOp>(op);
      builder.create<memref::StoreOp>(op->getLoc(), store.getValue(), store.getMemRef(), ValueRange{index});
    }
    op->erase();
  }
  bool hasLoop = false;
  function.walk([&](Operation *op) { hasLoop |= isa<scf::ForOp, affine::AffineForOp>(op); });
  if (!hasLoop) {
    // One zero-index store to the only one-element output identifies a
    // one-item computation. In partial mode it may be guarded, but must stay
    // on a nested true path. Wrapping in 0..1 changes no execution count.
    unsigned outputs = 0, stores = 0;
    for (const auto &object : objects) if (object.write) {
      ++outputs;
      if (object.elements != 1)
        return reject(function, "loop-free entry requires a one-element output");
    }
    function.walk([&](memref::StoreOp store) {
      auto arg = dyn_cast<BlockArgument>(store.getMemref());
      if (!arg || arg.getOwner() != &function.front() || !objects[arg.getArgNumber()].write ||
          store.getIndices().size() != 1 || constant(store.getIndices()[0]) != 0) return;
      bool partial = objects[arg.getArgNumber()].partial;
      if (!partial && store->getParentOp() != function.getOperation()) return;
      Operation *child = store;
      for (Operation *parent = store->getParentOp(); parent != function.getOperation();
           child = parent, parent = parent->getParentOp()) {
        auto branch = dyn_cast<scf::IfOp>(parent);
        if (!partial || !branch || child->getParentRegion() != &branch.getThenRegion()) return;
      }
      ++stores;
    });
    if (outputs != 1 || stores != 1)
      return reject(function, "loop-free entry requires one unconditional zero-indexed output store");
    OpBuilder builder(&function.front(), function.front().begin());
    auto single = builder.create<affine::AffineForOp>(function.getLoc(), 0, 1);
    auto &ops = function.front().getOperations();
    auto &body = single.getBody()->getOperations();
    body.splice(body.begin(), ops, std::next(Block::iterator(single)), std::prev(ops.end()));
    for (Operation &op : *single.getBody()) {
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        if (load.getIndices().size() == 1 && constant(load.getIndices()[0]) == 0)
          load.getIndicesMutable().assign(ValueRange{single.getInductionVar()});
      } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
        if (store.getIndices().size() == 1 && constant(store.getIndices()[0]) == 0)
          store.getIndicesMutable().assign(ValueRange{single.getInductionVar()});
      }
    }
    single.walk([&](memref::StoreOp store) {
      auto arg = dyn_cast<BlockArgument>(store.getMemref());
      if (arg && arg.getOwner() == &function.front() && objects[arg.getArgNumber()].partial &&
          store.getIndices().size() == 1 && constant(store.getIndices()[0]) == 0)
        store.getIndicesMutable().assign(ValueRange{single.getInductionVar()});
    });
  }
  Operation *loop = nullptr;
  Value iv;
  int64_t trips = 0;
  for (Operation &op : function.front()) {
    if (isa<arith::ConstantOp, func::ReturnOp>(op)) continue;
    if (loop) return reject(&op, "requires exactly one static stream loop");
    loop = &op;
    if (auto scfLoop = dyn_cast<scf::ForOp>(op)) {
      auto lo = constant(scfLoop.getLowerBound()), hi = constant(scfLoop.getUpperBound());
      auto step = constant(scfLoop.getStep());
      if (!lo || !hi || !step || *lo != 0 || *step != 1 || *hi <= 0 || scfLoop.getNumResults())
        return reject(loop, "requires static positive trip count, zero start, unit step and no carried state");
      trips = *hi; iv = scfLoop.getInductionVar();
    } else if (auto affineLoop = dyn_cast<affine::AffineForOp>(op)) {
      if (!affineLoop.hasConstantBounds() || affineLoop.getConstantLowerBound() != 0 ||
          affineLoop.getConstantUpperBound() <= 0 || affineLoop.getStep() != 1 || affineLoop.getNumResults())
        return reject(loop, "requires static positive trip count, zero start, unit step and no carried state");
      trips = affineLoop.getConstantUpperBound(); iv = affineLoop.getInductionVar();
    } else return reject(loop, "unsupported entry operation; expected static stream loop");
  }
  if (!loop) return reject(function, "requires one static positive stream loop");
  for (auto attr : loop->getAttrs())
    if (!isa<affine::AffineForOp>(loop) ||
        (attr.getName().strref() != "lowerBoundMap" &&
         attr.getName().strref() != "upperBoundMap" && attr.getName().strref() != "step" &&
         attr.getName().strref() != "operandSegmentSizes"))
      return reject(loop, "unsupported stream loop attributes");
  if (trips > 4095) return reject(loop, "trip count exceeds the fixed stream counter capacity");
  DenseMap<Value, unsigned> positions;
  for (auto arg : function.getArguments()) positions[arg] = arg.getArgNumber();
  SmallVector<memref::LoadOp> dynamicLoads;
  SmallVector<memref::StoreOp> conditionalStores;
  SmallVector<Operation *> linearMemory;
  SmallVector<arith::CmpIOp> comparisons;
  bool bad = false;
  auto error = [&](Operation *op, Twine message) { (void)reject(op, message); bad = true; };
  loop->walk([&](Operation *op) {
    if (op == loop) return;
    if (!isa<scf::IfOp, scf::YieldOp, affine::AffineYieldOp,
             arith::ConstantOp, arith::ExtSIOp, arith::ExtUIOp, arith::CmpIOp,
             arith::AndIOp, arith::IndexCastOp, arith::AddIOp,
             arith::MulIOp, memref::LoadOp, memref::StoreOp,
             affine::AffineLoadOp, affine::AffineStoreOp>(op)) {
      error(op, "unsupported effect or operation in static C stream"); return;
    }
    if (op->hasAttr("adora.no_access") || op->hasAttr("adora.require_no_access"))
      error(op, "unsupported memory attributes (no-access/volatile)");
    if (isa<arith::AddIOp, arith::MulIOp>(op))
      for (auto attr : op->getAttrs())
        if (attr.getName().strref() != "overflowFlags")
          error(op, "unsupported address arithmetic attributes");
    if (auto ext = dyn_cast<arith::ExtSIOp>(op))
      if (!ext.getIn().getType().isInteger(16) || !ext.getType().isInteger(32))
        error(op, "only i16 to i32 signed C promotions are supported");
    if (auto ext = dyn_cast<arith::ExtUIOp>(op))
      if (!allowPartialWrites || !ext.getIn().getType().isInteger(16) || !ext.getType().isInteger(32))
        error(op, "unsigned promotion requires partial-write i16 loaded-value zero comparison");
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      bool valueSelection = branch.getNumResults() == 1 &&
          branch.getResult(0).getType().isInteger(16) && !branch.getElseRegion().empty();
      bool conditionalWrite = allowPartialWrites && branch.getNumResults() == 0;
      if (!valueSelection && !conditionalWrite)
        error(op, "branch must select an i16 value with an explicit fallback");
    }
    if (auto cmp = dyn_cast<arith::CmpIOp>(op)) comparisons.push_back(cmp);
    Value memref; Value index; bool write = false;
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      memref = load.getMemref();
      if (load.getIndices().size() == 1) index = load.getIndices()[0];
      for (auto attr : op->getAttrs()) {
        auto flag = dyn_cast<BoolAttr>(attr.getValue());
        if (attr.getName().strref() != "nontemporal" || !flag || flag.getValue())
          error(op, "unsupported load attributes (no-access/volatile)");
      }
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      memref = store.getMemref(); write = true;
      if (store.getIndices().size() == 1) index = store.getIndices()[0];
      for (auto attr : op->getAttrs()) {
        auto flag = dyn_cast<BoolAttr>(attr.getValue());
        if (attr.getName().strref() != "nontemporal" || !flag || flag.getValue())
          error(op, "unsupported store attributes");
      }
    } else if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
      memref = load.getMemRef();
      if (load.getAffineMap().isIdentity() && load.getMapOperands().size() == 1)
        index = load.getMapOperands()[0];
      if (op->getAttrs().size() > 1) error(op, "unsupported affine load attributes");
    } else if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
      memref = store.getMemRef(); write = true;
      if (store.getAffineMap().isIdentity() && store.getMapOperands().size() == 1)
        index = store.getMapOperands()[0];
      if (op->getAttrs().size() > 1) error(op, "unsupported affine store attributes");
    }
    if (!memref) return;
    if (!positions.count(memref) || !index) { error(op, "memory must address a declared whole object"); return; }
    auto &object = objects[positions[memref]];
    if (write) ++object.writes; else ++object.reads;
    if (!write && object.partial)
      error(op, "explicit inout reads/feedback are unsupported; old values are preserved by initialization");
    if (write && op->getParentOp() != loop) {
      if (!allowPartialWrites || !object.partial) {
        error(op, "conditional write effects require an inout object and allow-partial-writes=true");
        return;
      }
      if (index != iv) { error(op, "conditional output must use the same iteration index"); return; }
      Operation *child = op;
      for (Operation *parent = op->getParentOp(); parent != loop;
           child = parent, parent = parent->getParentOp()) {
        auto branch = dyn_cast<scf::IfOp>(parent);
        if (!branch || branch.getNumResults() || child->getParentRegion() != &branch.getThenRegion()) {
          error(op, "conditional output requires a nested true guard path without results"); return;
        }
      }
      conditionalStores.push_back(cast<memref::StoreOp>(op));
      return;
    }
    if (index == iv && op->getParentOp() == loop) {
      if (object.elements < trips) error(op, "object length is smaller than the stream trip count");
      linearMemory.push_back(op);
    } else if (!write && isa<memref::LoadOp>(op) && op->getParentOfType<scf::IfOp>())
      dynamicLoads.push_back(cast<memref::LoadOp>(op));
    else error(op, "requires direct stream access or a guarded read; branch stores/dependent loads unsupported");
  });
  unsigned outputs = 0;
  for (auto &object : objects) {
    if (object.write) {
      ++outputs;
      if (object.partial) {
        if (object.reads || object.writes != 1 || conditionalStores.size() != 1 ||
            object.elements != trips || object.dtype != "uint16")
          error(function, "inout effects require one conditional uint16 output store, no explicit reads, and length equal to trip count");
      } else if (object.reads || object.writes != 1 || object.elements != trips)
        error(function, "write effects require one complete unconditional output stream");
    } else if (!object.reads || object.writes)
      error(function, "read effects contradict the object contract");
  }
  if (outputs != 1) error(function, "requires exactly one write object");
  if (bad) return failure();
  if (!conditionalStores.empty()) {
    if (dynamicLoads.size() != 1 || conditionalStores.front().getValue() != dynamicLoads.front().getResult())
      return reject(function, "conditional output requires exactly one guarded source loaded value");
    const auto &source = objects[positions[dynamicLoads.front().getMemref()]];
    if (source.write || source.dtype != "uint16" || source.reads != 1)
      return reject(function, "conditional source must be a single read-only uint16 loaded value");
  }

  // Comparison narrowing is exact only for sign extensions and representable
  // signed constants. Unsigned promoted comparisons and arbitrary i32 reject.
  for (auto cmp : comparisons) {
    auto type = cmp.getLhs().getType();
    if (!type.isInteger(32) && !type.isInteger(16))
      return reject(cmp, "only signed i16 promotion comparisons are supported");
    using P = arith::CmpIPredicate;
    auto pred = cmp.getPredicate();
    if (pred == P::ult || pred == P::ule || pred == P::ugt || pred == P::uge)
      return reject(cmp, "unsigned comparison of promoted coordinates is unsupported");
    OpBuilder builder(cmp);
    Value lhs = signedLeaf(cmp.getLhs()), rhs = signedLeaf(cmp.getRhs());
    // C promotes an unsigned-short pixel to i32 with extui. Equality with
    // zero is bit-exact at either width, including 0x8000 and 0xffff. This does
    // not authorize signed ordering, general unsigned comparisons or addresses.
    auto narrowUnsignedZero = [&](Value value, Value other) -> Value {
      auto ext = value.getDefiningOp<arith::ExtUIOp>();
      if (!ext) return value;
      auto load = ext.getIn().getDefiningOp<memref::LoadOp>();
      if (!allowPartialWrites || (pred != P::eq && pred != P::ne) || constant(other) != 0 ||
          !load || !llvm::is_contained(dynamicLoads, load) ||
          objects[positions[load.getMemref()]].dtype != "uint16") return {};
      return ext.getIn();
    };
    lhs = narrowUnsignedZero(lhs, rhs);
    rhs = narrowUnsignedZero(rhs, cmp.getLhs());
    if (!lhs || !rhs) return reject(cmp, "unsigned pixel promotion only supports an exact zero comparison");
    auto narrow = [&](Value value) -> Value {
      if (value.getType().isInteger(16)) return value;
      auto c = constant(value);
      if (!c || *c < -32768 || *c > 32767) return {};
      return builder.create<arith::ConstantIntOp>(cmp.getLoc(), *c, 16);
    };
    lhs = narrow(lhs); rhs = narrow(rhs);
    if (!lhs || !rhs) return reject(cmp, "comparison is not an exact signed i16 promotion");
    // Canonicalize constant-left and strict-lower comparisons for the shared
    // R003 guard proof, without relying on frontend text or variable names.
    if (constant(lhs) && !constant(rhs)) { std::swap(lhs, rhs); pred = reverse(pred); }
    if (pred == P::sgt) {
      auto c = constant(rhs);
      if (!c || *c == 32767) return reject(cmp, "unsupported strict signed lower guard");
      rhs = builder.create<arith::ConstantIntOp>(cmp.getLoc(), *c + 1, 16);
      pred = P::sge;
    }
    auto replacement = builder.create<arith::CmpIOp>(cmp.getLoc(), pred, lhs, rhs);
    cmp.replaceAllUsesWith(replacement.getResult()); cmp.erase();
  }

  for (auto load : dynamicLoads) {
    SmallVector<Value> guards;
    Operation *child = load;
    for (Operation *parent = load->getParentOp(); parent != loop; child = parent, parent = parent->getParentOp()) {
      auto branch = dyn_cast<scf::IfOp>(parent);
      if (!branch || child->getParentRegion() != &branch.getThenRegion())
        return reject(load, "dynamic reads require a proven true guard path");
      guards.push_back(branch.getCondition());
    }
    OpBuilder builder(load);
    unsigned proofSteps = 0;
    std::function<std::optional<Range>(Value, unsigned)> prove;
    prove = [&](Value value, unsigned depth) -> std::optional<Range> {
      if (depth > 64 || ++proofSteps > 256) return std::nullopt;
      if (auto c = constant(value)) {
        if (*c < 0 || *c > 32767) return std::nullopt;
        return Range{*c, *c};
      }
      if (auto cast = value.getDefiningOp<arith::IndexCastOp>())
        return prove(cast.getIn(), depth + 1);
      Value leaf = signedLeaf(value);
      if (leaf.getType().isInteger(16)) {
        auto *definition = leaf.getDefiningOp();
        if (!definition || definition->getParentOp() != loop ||
            !isa<memref::LoadOp, affine::AffineLoadOp>(definition)) return std::nullopt;
        Value object = isa<memref::LoadOp>(definition)
            ? cast<memref::LoadOp>(definition).getMemref()
            : cast<affine::AffineLoadOp>(definition).getMemRef();
        if (objects[positions[object]].dtype != "int16") return std::nullopt;
        Range range{-32768, 32767};
        DenseSet<Value> visited;
        for (Value guard : guards) refine(guard, leaf, range, visited);
        if (range.first < 0 || range.second < range.first) return std::nullopt;
        return range;
      }
      auto *op = value.getDefiningOp();
      if (!op || !isa<arith::AddIOp, arith::MulIOp>(op) ||
          (!value.getType().isIndex() && !value.getType().isInteger(32))) return std::nullopt;
      auto lhs = prove(op->getOperand(0), depth + 1), rhs = prove(op->getOperand(1), depth + 1);
      if (!lhs || !rhs) return std::nullopt;
      Range result = isa<arith::AddIOp>(op)
          ? Range{lhs->first + rhs->first, lhs->second + rhs->second}
          : Range{lhs->first * rhs->first, lhs->second * rhs->second};
      // All operands are nonnegative and every intermediate fits signed i16;
      // hence both i32 nsw and nuw are proven, not discarded speculatively.
      if (result.second > 32767) return std::nullopt;
      return result;
    };
    auto address = load.getIndices()[0];
    auto range = prove(address, 0);
    if (!range || range->second >= objects[positions[load.getMemref()]].elements || range->second * 2 > 32767)
      return reject(load, "guard-linked address proof failed: intermediate/byte address or object bounds unsafe");
    std::function<Value(Value)> rebuild = [&](Value value) -> Value {
      if (auto c = constant(value)) return builder.create<arith::ConstantIndexOp>(load.getLoc(), *c);
      if (auto cast = value.getDefiningOp<arith::IndexCastOp>()) return rebuild(cast.getIn());
      Value leaf = signedLeaf(value);
      if (leaf.getType().isInteger(16))
        return builder.create<arith::IndexCastOp>(load.getLoc(), builder.getIndexType(), leaf);
      auto *op = value.getDefiningOp();
      Value lhs = rebuild(op->getOperand(0)), rhs = rebuild(op->getOperand(1));
      if (isa<arith::AddIOp>(op)) return builder.create<arith::AddIOp>(load.getLoc(), lhs, rhs);
      return builder.create<arith::MulIOp>(load.getLoc(), lhs, rhs);
    };
    load.getIndicesMutable().assign(ValueRange{rebuild(address)});
  }
  // Erase only dead, pure promotion/address operations after reconstructing
  // their uses. Any remaining wide computation is outside this C subset.
  bool changed;
  do {
    changed = false;
    SmallVector<Operation *> dead;
    function.walk([&](Operation *op) {
      if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::IndexCastOp, arith::AddIOp, arith::MulIOp, arith::ConstantOp>(op) &&
          op->use_empty()) dead.push_back(op);
    });
    for (auto *op : llvm::reverse(dead)) { op->erase(); changed = true; }
  } while (changed);
  function.walk([&](Operation *op) {
    for (Type type : op->getResultTypes())
      if (type.isInteger(32)) error(op, "unproven i32 computation remains after C promotion normalization");
    if (isa<arith::AddIOp, arith::MulIOp>(op) && !op->getResult(0).getType().isIndex())
      error(op, "only proven address arithmetic is supported");
  });
  if (bad) return failure();

  OpBuilder builder(function.getContext());
  SmallVector<Type> types;
  for (auto arg : function.getArguments()) {
    auto type = MemRefType::get({objects[arg.getArgNumber()].elements}, builder.getI16Type());
    arg.setType(type); types.push_back(type);
  }
  function.setType(builder.getFunctionType(types, {}));
  std::string kernelName = function.getSymName().str();
  builder.setInsertionPoint(loop);
  SmallVector<Value> local;
  auto zeroMap = AffineMap::getConstantMap(0, function.getContext());
  for (auto arg : function.getArguments()) {
    auto &object = objects[arg.getArgNumber()];
    auto type = cast<MemRefType>(arg.getType());
    Operation *transfer;
    if (object.write && !object.partial)
      transfer = builder.create<LocalMemAllocOp>(function.getLoc(), type, kernelName);
    else
      transfer = builder.create<DataBlockLoadOp>(function.getLoc(), arg, zeroMap, ValueRange{}, type, kernelName);
    transfer->setAttr("Id", builder.getStringAttr("arg" + std::to_string(arg.getArgNumber())));
    transfer->setAttr("adora.host_dtype", builder.getStringAttr(object.dtype));
    local.push_back(transfer->getResult(0));
    arg.replaceUsesWithIf(local.back(), [&](OpOperand &use) { return loop->isAncestor(use.getOwner()); });
  }
  for (Operation *op : linearMemory) {
    builder.setInsertionPoint(op);
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      auto replacement = builder.create<affine::AffineLoadOp>(op->getLoc(), load.getMemref(), ValueRange{iv});
      load.replaceAllUsesWith(replacement.getResult()); load.erase();
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      builder.create<affine::AffineStoreOp>(op->getLoc(), store.getValue(), store.getMemref(), ValueRange{iv});
      store.erase();
    }
  }
  if (auto old = dyn_cast<scf::ForOp>(loop)) {
    builder.setInsertionPoint(old);
    auto replacement = builder.create<affine::AffineForOp>(old.getLoc(), 0, trips);
    old.getInductionVar().replaceAllUsesWith(replacement.getInductionVar());
    auto &ops = old.getBody()->getOperations();
    replacement.getBody()->getOperations().splice(replacement.getBody()->begin(), ops, ops.begin(), std::prev(ops.end()));
    loop = replacement; old.erase();
  }
  builder.setInsertionPoint(loop);
  auto kernel = builder.create<KernelOp>(function.getLoc());
  kernel.setKernelName(kernelName);
  builder.setInsertionPointToEnd(&kernel.getBody().front());
  auto terminator = builder.create<TerminatorOp>(function.getLoc());
  loop->moveBefore(terminator);
  builder.setInsertionPointAfter(kernel);
  for (auto arg : function.getArguments()) {
    unsigned pos = arg.getArgNumber();
    if (!objects[pos].write) continue;
    auto store = builder.create<DataBlockStoreOp>(function.getLoc(), local[pos], arg, zeroMap, ValueRange{}, kernelName);
    store->setAttr("Id", builder.getStringAttr("arg" + std::to_string(pos)));
  }
  function->setAttr("adora.c_entry", builder.getUnitAttr());
  return success();
}

#include "LowerLargeCEntry.inc"

std::unique_ptr<OperationPass<ModuleOp>> mlir::ADORA::createLowerCEntryPass() {
  return std::make_unique<LowerCEntryPass>();
}
