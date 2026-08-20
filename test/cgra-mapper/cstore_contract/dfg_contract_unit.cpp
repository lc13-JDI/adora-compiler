#include "ir/cstore_contract.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: cstore-dfg-contract-test <case>\n";
    return 2;
  }

  const std::string testCase = argv[1];
  // These represent raw incoming CDFG edges.  A constant's edge is still
  // recorded even though DFGIR subsequently folds that constant into an
  // immediate; a memory dependence is intentionally ignored.
  struct RawEdge {
    int logicalPort;
    bool isMemory;
  };
  std::vector<RawEdge> edges;
  if (testCase == "valid")
    edges = {{0, false}, {1, false}, {2, false}};
  else if (testCase == "missing")
    edges = {{0, false}, {1, false}};
  else if (testCase == "duplicate")
    edges = {{0, false}, {0, false}, {1, false}, {2, false}};
  else if (testCase == "out-of-range")
    edges = {{0, false}, {1, false}, {3, false}};
  else if (testCase == "memory-ignored")
    edges = {{0, false}, {1, false}, {2, false}, {2, true}};
  else {
    std::cerr << "unknown case: " << testCase << "\n";
    return 2;
  }

  std::map<int, int> ports;
  for (const RawEdge &edge : edges)
    CStoreContract::recordNonMemoryLogicalPort(ports, edge.logicalPort,
                                                edge.isMemory);
  std::string violation = CStoreContract::logicalPortViolation(ports);
  if (violation.empty())
    return 0;
  std::cout << "Invalid CSTORE DFG node unit: " << violation << "\n";
  return 1;
}
