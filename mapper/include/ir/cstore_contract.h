#ifndef __CSTORE_CONTRACT_H__
#define __CSTORE_CONTRACT_H__

#include <map>
#include <string>

namespace CStoreContract {

inline void recordNonMemoryLogicalPort(std::map<int, int> &portUse,
                                       int logicalPort, bool isMemory) {
  if (!isMemory)
    ++portUse[logicalPort];
}

inline std::string logicalPortViolation(const std::map<int, int> &portUse) {
  for (const auto &use : portUse) {
    if (use.first < 0 || use.first > 2)
      return "logical operand " + std::to_string(use.first) +
             " is out of range";
  }
  for (int operand = 0; operand < 3; ++operand) {
    auto use = portUse.find(operand);
    if (use == portUse.end() || use->second != 1)
      return "logical operand " + std::to_string(operand) +
             " must be covered exactly once";
  }
  return "";
}

} // namespace CStoreContract

#endif
