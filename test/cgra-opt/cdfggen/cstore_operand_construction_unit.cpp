// Every candidate edge goes through the production construction function.
#include "mlir_cdfg.h"
#include "cstore_ports.h"
#include <algorithm>
#include <iostream>
#include <set>

static bool keys(LLVMCDFGNode *node, std::initializer_list<int> wanted) {
  std::set<int> actual;
  for (const auto &entry : node->inputInfoMap())
    for (const auto &info : entry.second)
      if (!actual.insert(info.idx).second) return false;
  return actual == std::set<int>(wanted);
}

int main() {
  LLVMCDFG g("operand-construction");
  auto node = [&](const char *type) {
    auto *n = g.addNode(type); n->setTypeName(type); return n;
  };
  auto *step = node("CONST"), *acc = node("ACC"), *size = node("CONST");
  auto *mul = node("MUL"), *sink = node("ADD"), *explicitStore = node("CSTORE");
  auto *store = node("CSTORE"), *value = node("Input"), *pred = node("Input");
  acc->setAcc(); acc->setLoopIndexAcc();
  acc->setACCcount("4"); acc->setACCinterval("1"); acc->setACCrepeat("1");
  store->setSequentialCStore(4, "kernel:dst");
  g.addOperandEdge(step, acc, 0);
  g.addOperandEdge(acc, sink, 0);
  g.addOperandEdge(acc, mul, 0);
  g.addOperandEdge(size, mul, 1);
  g.addOperandEdge(mul, explicitStore, 1);
  g.addOperandEdge(acc, store, 1); // no edge, shared index must survive
  g.addOperandEdge(mul, store, 1); // no edge, shared byte address must survive
  g.addOperandEdge(pred, store, 2); // deliberately reversed enumeration
  g.addOperandEdge(value, store, 0);
  if (!keys(store, {0,2}) || !keys(acc, {0}) || !keys(mul, {0,1}) ||
      !keys(explicitStore, {1}) || !g.edge(step, acc) || !g.edge(acc, sink) ||
      !g.edge(mul, explicitStore) || g.edge(acc, store) || g.edge(mul, store))
    return 1;
  auto *same = node("CSTORE"); same->setSequentialCStore(4, "kernel:dst2");
  g.addOperandEdge(value, same, 2);
  g.addOperandEdge(value, same, 0);
  g.addOperandEdge(value, same, 1);
  if (!keys(same, {0,2}) || same->inputNodes().size() != 1 ||
      same->inputEdges().size() != 2 ||
      same->getInputIndices(value) != std::vector<int>({0,2})) return 2;
  auto *ordinary = node("CSTORE");
  g.addOperandEdge(value, ordinary, 0);
  g.addOperandEdge(acc, ordinary, 1);
  g.addOperandEdge(pred, ordinary, 2);
  if (!keys(ordinary, {0,1,2}) || !g.edge(acc, ordinary)) return 3;
  // A producer used exclusively as a proposed address never acquires an edge
  // or an adjacency entry. No graph-pruning algorithm exists in this test.
  auto *exclusiveIndex = node("for"), *exclusive = node("CSTORE");
  exclusive->setSequentialCStore(4, "kernel:exclusive");
  g.addOperandEdge(exclusiveIndex, exclusive, 1);
  g.addOperandEdge(value, exclusive, 0);
  g.addOperandEdge(pred, exclusive, 2);
  if (!exclusiveIndex->outputNodes().empty() ||
      !exclusiveIndex->outputEdges().empty() || !keys(exclusive, {0,2})) return 4;
  // Use the same production checker as graph generation and mapper ingest.
  // Sparse edges alone, or a wrong mode, never authorize sequential operation.
  using CStoreContract::cstoreLogicalPortViolation;
  if (!cstoreLogicalPortViolation({{0,1},{2,1}}, true).empty() ||
      cstoreLogicalPortViolation({{0,1},{2,1}}, false).empty() ||
      cstoreLogicalPortViolation({{0,1},{1,1}}, true).empty() ||
      cstoreLogicalPortViolation({{0,1},{2,2}}, true).empty() ||
      cstoreLogicalPortViolation({{0,1},{1,1},{2,1}}, true).empty() ||
      !cstoreLogicalPortViolation({{0,1},{1,1},{2,1}}, false).empty() ||
      !cstoreLogicalPortViolation({{0,1},{1,1},{2,1}}, true, true).empty() ||
      cstoreLogicalPortViolation({{0,1},{2,1},{3,1}}, true).empty() ||
      cstoreLogicalPortViolation({{-1,1},{0,1},{2,1}}, true).empty()) return 5;
  // Backedges and memory dependencies are never elided by data construction.
  auto *backedgeStore = node("CSTORE");
  backedgeStore->setSequentialCStore(4, "kernel:backedge");
  auto *backedge = g.addOperandEdge(acc, backedgeStore, 1, true);
  if (!backedge || !keys(backedgeStore, {1}) ||
      !backedgeStore->isInputBackEdge(acc)) return 6;
  auto *memory = g.addEdge(sink, store, EDGE_TYPE_MEM);
  if (!memory || memory->type() != EDGE_TYPE_MEM ||
      !g.edge(sink, store)) return 7;
  std::cout << "shared index/step, shared byte MUL, exact sparse keys and explicit address preserved\n";
  return 0;
}
