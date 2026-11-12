#ifndef ADORA_CSTORE_PORTS_H
#define ADORA_CSTORE_PORTS_H

#include <map>
#include <string>

namespace CStoreContract {

// Logical ports retain their hardware identities even when address generation
// is internal. Legacy three-edge records may be accepted only after the caller
// has independently validated their sequential proof.
inline std::string cstoreLogicalPortViolation(
    const std::map<int, int> &portUse, bool trustedSequential,
    bool allowLegacyAddress = false) {
  for (const auto &use : portUse) {
    if (use.first < 0 || use.first > 2)
      return "logical operand " + std::to_string(use.first) + " is out of range";
    if (use.second != 1)
      return "logical operand " + std::to_string(use.first) +
             " must be covered exactly once";
  }
  for (int port : {0, 2}) {
    if (!portUse.count(port))
      return "logical operand " + std::to_string(port) +
             " must be covered exactly once";
  }
  if (!trustedSequential && !portUse.count(1))
    return "logical operand 1 must be covered exactly once";
  if (trustedSequential && !allowLegacyAddress && portUse.count(1))
    return "sequential CSTORE must not connect logical operand 1";
  return "";
}

} // namespace CStoreContract

#endif
