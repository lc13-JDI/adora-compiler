#include "ir/dfg_ir.h"
#include "mlir_cdfg.h"
#include "op/operations.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"

namespace {

LLVMCDFGNode *addNode(LLVMCDFG &cdfg, mlir::MLIRContext &context,
                      const std::string &type) {
  mlir::OperationState state(mlir::UnknownLoc::get(&context), "test.node");
  return cdfg.addNode(mlir::Operation::create(state), type);
}

void addEdge(LLVMCDFG &cdfg, LLVMCDFGNode *source, LLVMCDFGNode *destination,
             int logicalPort, EdgeType type = EDGE_TYPE_DATA) {
  destination->addInputNode(source, logicalPort);
  source->addOutputNode(destination);
  cdfg.addEdge(source, destination, type);
}

// Exercise the post-ingest graph, including logical ports after immediates
// were folded and transparent PASS nodes inserted by routing.
bool recognizesPrefix(const std::string &variant) {
  DFG graph;
  graph.setId(0);
  int nodeId = 0, edgeId = 0;
  auto node = [&](DFGNode *n, const char *op) {
    n->setId(++nodeId);
    n->setOperation(op);
    graph.addNode(n);
    return n;
  };
  auto edge = [&](DFGNode *src, DFGNode *dst, int port) {
    auto *e = new DFGEdge(++edgeId);
    e->setEdge(src->id(), 0, dst->id(), port);
    graph.addEdge(e);
  };
  auto io = [&](const char *op, const char *object, int stride) {
    auto *n = new DFGIONode;
    node(n, op);
    graph.addIONode(n->id());
    n->setMemSize(16);
    n->setMemRefName(object);
    n->addPatternLevel(stride, 8);
    return n;
  };
  auto binary = [&](const char *op, DFGNode *lhs, DFGNode *rhs) {
    auto *n = node(new DFGNode, op);
    edge(lhs, n, 0);
    edge(rhs, n, 1);
    return n;
  };
  auto immediate = [&](const char *op, DFGNode *source, int constPort, int value) {
    auto *n = node(new DFGNode, op);
    edge(source, n, 1 - constPort);
    n->setImmIdx(constPort);
    n->setImm(value);
    return n;
  };
  auto *mode = io("INPUT", "z", 2), *idx = io("INPUT", "y", 2);
  auto *load = io("LOAD", "x", variant == "load-stride" ? 2 : 0);
  auto *store = io("CSTORE", variant == "same-object" ? "x" : "w", 2);
  if (variant != "explicit-store") store->setSequentialCStore(8, store->memRefName());
  auto *negative = immediate("SLT", mode, 1, 0);
  auto *nonzero = binary("OR", immediate("SLT", mode, 0, 0), negative);
  auto *lower = immediate("SLE", idx, 0, 0);
  auto *upper = immediate("SLT", idx, 1, variant == "bad-bound" ? 9 : 8);
  auto *guard = binary("AND", upper, binary("AND", nonzero, lower));
  auto *select = node(new DFGNode, "SEL");
  select->setImmIdx(0);
  select->setImm(variant == "unsafe-fallback" ? 8 : 0);
  edge(idx, select, 1);
  edge(variant == "unsafe-load" ? nonzero : guard, select, 2);
  edge(immediate("MUL", select, 0, 2), load, 0);
  auto *acc = node(new DFGNode, "ACC");
  acc->setAccumulative(true);
  acc->setInitVal(0);
  acc->setInterval(1);
  acc->setCycles(variant == "short-count" ? 7 : 8);
  acc->setRepeats(1);
  acc->setIsAccFirst(variant != "exclusive");
  if (variant == "index-acc") acc->setLoopIndexAcc();
  auto *sign = variant == "wrong-mode" ? immediate("SLT", idx, 1, 0) : negative;
  edge(variant == "wrong-sign" ? immediate("SLT", mode, 0, 0) : sign, acc, 0);
  auto *alive = immediate("SLE", acc, 1, variant == "late-stop" ? 1 : 0);
  auto *gate = binary("AND", alive, guard);
  auto *pass = node(new DFGNode, "PASS");
  edge(variant == "missing-prefix" ? guard : gate, pass, 0);
  edge(pass, store, 2);
  edge(variant == "other-payload" ? mode : load, store, 0);
  return graph.boundedBreakPrefixAcc() == acc;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3 && argc != 4) {
    std::cerr << "usage: cstore-dfg-contract-test <case> <operations.json>\n";
    return 2;
  }

  const std::string testCase = argv[1];
  if (testCase == "bounded-break-prefix") {
    Operations::Instance(argv[2]);
    for (const char *variant : {"valid", "short-count", "exclusive", "index-acc",
                                "late-stop", "missing-prefix", "wrong-mode", "wrong-sign",
                                "unsafe-load", "unsafe-fallback", "bad-bound", "load-stride",
                                "same-object", "explicit-store", "other-payload"}) {
      if (recognizesPrefix(variant) != (std::string(variant) == "valid")) {
        std::cerr << "unexpected break-prefix classification for " << variant << "\n";
        return 1;
      }
    }
    return 0;
  }
  if (testCase == "json" && argc == 4) {
    Operations::Instance(argv[2]);
    DFGIR dfg(argv[3]);
    auto *io = dynamic_cast<DFGIONode *>(dfg.getDFG()->node(1));
    return io && !io->isSequentialCStore() ? 0 : 2;
  }
  if (testCase.rfind("forged-sequential", 0) == 0) {
    Operations::Instance(argv[2]);
    mlir::MLIRContext context;
    context.allowUnregisteredDialects();
    LLVMCDFG cdfg("forged-sequential");
    auto *store = addNode(cdfg, context, "CSTORE");
    if (testCase == "forged-sequential-missing-mode")
      store->operation()->setAttr("adora.cstore_address_mode", mlir::StringAttr::get(&context, "sequential"));
    else
      store->setSequentialCStore(testCase == "forged-sequential-zero" ? 0 : 4, "kernel:dst");
    store->setMemrefName("kernel:dst");
    store->setMemrefSize(8);
    store->setInitAddr("0");
    store->setLinearAccess("2,4");
    for (int port = 0; port < 3; ++port)
      addEdge(cdfg, addNode(cdfg, context, "Input"), store, port);
    DFGIR dfg(&cdfg);
    return dfg.getDFG() ? 0 : 2;
  }
  if (testCase.rfind("sequential-record-", 0) == 0) {
    Operations::Instance(argv[2]);
    mlir::MLIRContext context;
    context.allowUnregisteredDialects();
    context.getOrLoadDialect<mlir::ADORA::ADORADialect>();
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    mlir::OperationState valuesState(loc, "test.values");
    valuesState.addTypes({builder.getI16Type(),
                         mlir::MemRefType::get({4}, builder.getI16Type()),
                         builder.getIndexType(), builder.getI1Type()});
    auto *values = mlir::Operation::create(valuesState);
    mlir::OperationState state(loc, mlir::ADORA::CondStoreOp::getOperationName());
    state.addOperands(values->getResults());
    state.addAttribute("adora.cstore_address_mode", builder.getStringAttr("sequential"));
    state.addAttribute("adora.cstore_slots", builder.getI64IntegerAttr(4));
    state.addAttribute("adora.cstore_object", builder.getStringAttr("kernel:dst"));
    auto *operation = mlir::Operation::create(state);
    LLVMCDFG cdfg("sequential-record");
    auto *store = cdfg.addNode(operation, "CSTORE");
    store->setSequentialCStore(4, "kernel:dst");
    store->setMemrefName("kernel:dst");
    store->setMemrefSize(8);
    store->setInitAddr("0");
    store->setLinearAccess("2,4");
    if (testCase == "sequential-record-zero") store->setSequentialCStore(0, "kernel:dst");
    if (testCase == "sequential-record-overflow") store->setSequentialCStore(4096, "kernel:dst");
    if (testCase == "sequential-record-count") operation->setAttr("adora.cstore_slots", builder.getI64IntegerAttr(3));
    if (testCase == "sequential-record-count-type") operation->setAttr("adora.cstore_slots", builder.getI32IntegerAttr(4));
    if (testCase == "sequential-record-source-object") operation->setAttr("adora.cstore_object", builder.getStringAttr("kernel:src"));
    if (testCase == "sequential-record-node-object") store->setSequentialCStore(4, "kernel:src");
    if (testCase == "sequential-record-mode") operation->setAttr("adora.cstore_address_mode", builder.getStringAttr("explicit"));
    if (testCase == "sequential-record-pattern") store->setLinearAccess("0,4");
    if (testCase == "sequential-record-missing") operation->removeAttr("adora.cstore_slots");
    if (testCase == "sequential-record-size") store->setMemrefSize(6);
    if (testCase == "sequential-record-offset") store->setInitAddr("2");
    std::vector<int> ports{0, 1, 2};
    if (testCase == "sequential-record-sparse") ports = {0, 2};
    if (testCase == "sequential-record-sparse-reordered") ports = {2, 0};
    if (testCase == "sequential-record-sparse-shared") ports = {0, 2};
    if (testCase == "sequential-record-sparse-predicate-constant") ports = {2, 0};
    if (testCase == "sequential-record-sparse-value-constant") ports = {0, 2};
    if (testCase == "sequential-record-sparse-wrong") ports = {0, 1};
    if (testCase == "sequential-record-sparse-missing") ports = {0};
    if (testCase == "sequential-record-sparse-duplicate") ports = {0, 2, 2};
    auto *shared = addNode(cdfg, context, "Input");
    std::map<int, int> expectedSourceIds;
    for (int port : ports) {
      bool constant = (port == 2 && testCase == "sequential-record-sparse-predicate-constant") ||
                      (port == 0 && testCase == "sequential-record-sparse-value-constant");
      auto *source = testCase == "sequential-record-sparse-shared" ? shared :
                     addNode(cdfg, context, constant ? "CONST" : "Input");
      if (!constant) expectedSourceIds[port] = source->id() + 1;
      addEdge(cdfg, source, store, port);
    }
    DFGIR dfg(&cdfg);
    auto *io = dynamic_cast<DFGIONode *>(dfg.getDFG()->node(store->id() + 1));
    if (!io || !io->isSequentialCStore() || io->sequentialCStoreSlots() != 4) return 2;
    if (ports.size() == 2 && (io->inputEdges().size() != 2 || !io->inputEdges().count(0) || !io->inputEdges().count(2))) return 3;
    // Preserve source identity independently of sparse-edge enumeration order.
    for (const auto &expected : expectedSourceIds) {
      auto edge = io->inputEdges().find(expected.first);
      if (edge == io->inputEdges().end() ||
          dfg.getDFG()->edge(edge->second)->srcId() != expected.second) return 6;
    }
    if (testCase == "sequential-record-sparse-shared" &&
        dfg.getDFG()->edge(io->inputEdges().at(0))->srcId() != dfg.getDFG()->edge(io->inputEdges().at(2))->srcId()) return 4;
    if (testCase.find("constant") != std::string::npos) {
      int port = testCase == "sequential-record-sparse-value-constant" ? 0 : 2;
      auto *pass = dfg.getDFG()->node(dfg.getDFG()->edge(io->inputEdges().at(port))->srcId());
      if (pass->operation() != "PASS" || !pass->hasImm()) return 5;
    }
    return 0;
  }
  if (testCase.rfind("load-", 0) == 0) {
    Operations::Instance(argv[2]);
    mlir::MLIRContext context;
    context.allowUnregisteredDialects();
    LLVMCDFG cdfg("load-contract-unit");
    auto *load = addNode(cdfg, context, "load");
    load->setMemrefName(testCase == "load-no-object" ? "" : "kernel:w");
    load->setMemrefSize(testCase == "load-no-size" ? 0 : 16);
    load->setInitAddr("0");
    load->setLinearAccess("0,1");
    if (testCase == "load-loop") load->setLinearAccess("0,4");
    if (testCase == "load-empty") load->setLinearAccess("0,0");
    if (testCase == "load-negative") load->setLinearAccess("0,-1");
    if (testCase == "load-nonzero-stride") load->setLinearAccess("2,4");
    if (testCase == "load-overflow") load->setLinearAccess("0,2147483648");
    if (testCase != "load-missing-port")
      addEdge(cdfg, addNode(cdfg, context, "Input"), load,
              testCase == "load-wrong-port" ? 1 : 0);
    if (testCase == "load-duplicate-port")
      addEdge(cdfg, addNode(cdfg, context, "Input"), load, 0);
    DFGIR dfg(&cdfg);
    return dfg.getDFG() ? 0 : 2;
  }
  const bool isStore = testCase.rfind("store-", 0) == 0;
  const std::string shape = isStore ? testCase.substr(6) : testCase;
  if (shape != "zero-immediate" && shape != "one-immediate" &&
      shape != "two-immediate" && shape != "missing" &&
      shape != "duplicate" && shape != "out-of-range" &&
      shape != "memory-ignored") {
    std::cerr << "unknown case: " << testCase << "\n";
    return 2;
  }

  Operations::Instance(argv[2]);
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  LLVMCDFG cdfg("contract-unit");
  LLVMCDFGNode *destination = addNode(cdfg, context, isStore ? "store" : "CSTORE");
  const int operandCount = isStore ? 2 : 3;
  for (int operand = 0; operand < operandCount; ++operand) {
    if (shape == "missing" && operand == operandCount - 1)
      continue;
    const bool immediate =
        (shape == "one-immediate" && operand == operandCount - 1) ||
        (shape == "two-immediate" && operand >= operandCount - 2);
    LLVMCDFGNode *source =
        addNode(cdfg, context, immediate ? "CONST" : "Input");
    int logicalPort = operand;
    if (shape == "duplicate" && operand == operandCount - 1)
      logicalPort = 0;
    if (shape == "out-of-range" && operand == operandCount - 1)
      logicalPort = operandCount;
    addEdge(cdfg, source, destination, logicalPort);
  }
  if (shape == "duplicate") {
    LLVMCDFGNode *last = addNode(cdfg, context, "Input");
    addEdge(cdfg, last, destination, operandCount - 1);
  }
  if (shape == "memory-ignored") {
    LLVMCDFGNode *memorySource = addNode(cdfg, context, "Input");
    addEdge(cdfg, memorySource, destination, operandCount - 1, EDGE_TYPE_MEM);
  }

  DFGIR dfg(&cdfg);
  return dfg.getDFG() ? 0 : 2;
}
