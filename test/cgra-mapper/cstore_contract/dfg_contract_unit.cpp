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
  if (argc != 3) {
    std::cerr << "usage: cstore-dfg-contract-test <case> <operations.json>\n";
    return 2;
  }

  const std::string testCase = argv[1];
  if (testCase != "valid" && testCase != "missing" &&
      testCase != "duplicate" && testCase != "out-of-range" &&
      testCase != "memory-ignored") {
    std::cerr << "unknown case: " << testCase << "\n";
    return 2;
  }

  Operations::Instance(argv[2]);
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  LLVMCDFG cdfg("contract-unit");
  LLVMCDFGNode *cstore = addNode(cdfg, context, "CSTORE");
  LLVMCDFGNode *first = addNode(cdfg, context, "CONST");
  LLVMCDFGNode *second = addNode(cdfg, context, "CONST");
  LLVMCDFGNode *third = addNode(cdfg, context, "CONST");

  addEdge(cdfg, first, cstore, 0);
  addEdge(cdfg, second, cstore, testCase == "duplicate" ? 0 : 1);
  if (testCase != "missing")
    addEdge(cdfg, third, cstore, testCase == "out-of-range" ? 3 : 2);
  if (testCase == "duplicate") {
    LLVMCDFGNode *fourth = addNode(cdfg, context, "CONST");
    addEdge(cdfg, fourth, cstore, 2);
  }
  if (testCase == "memory-ignored") {
    LLVMCDFGNode *memorySource = addNode(cdfg, context, "CONST");
    addEdge(cdfg, memorySource, cstore, 2, EDGE_TYPE_MEM);
  }

  DFGIR dfg(&cdfg);
  return dfg.getDFG() ? 0 : 2;
}
