#include "ir/dfg_ir.h"
#include "mlir_cdfg.h"
#include "op/operations.h"

#include <iostream>
#include <string>

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

} // namespace

int main(int argc, char **argv) {
  if (argc != 3 && argc != 4) {
    std::cerr << "usage: cstore-dfg-contract-test <case> <operations.json>\n";
    return 2;
  }

  const std::string testCase = argv[1];
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
    for (int port = 0; port < 3; ++port)
      addEdge(cdfg, addNode(cdfg, context, "Input"), store, port);
    DFGIR dfg(&cdfg);
    auto *io = dynamic_cast<DFGIONode *>(dfg.getDFG()->node(store->id() + 1));
    return io && io->isSequentialCStore() && io->sequentialCStoreSlots() == 4 ? 0 : 2;
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
