// RUN: cstore-dfg-contract-test load-loop %S/../../spec/cgra_cstore_vitra/artifacts/operations.json
// RUN: not cstore-dfg-contract-test load-empty %S/../../spec/cgra_cstore_vitra/artifacts/operations.json 2>&1 | %FileCheck %s
// RUN: not cstore-dfg-contract-test load-negative %S/../../spec/cgra_cstore_vitra/artifacts/operations.json 2>&1 | %FileCheck %s
// RUN: not cstore-dfg-contract-test load-nonzero-stride %S/../../spec/cgra_cstore_vitra/artifacts/operations.json 2>&1 | %FileCheck %s
// RUN: not cstore-dfg-contract-test load-overflow %S/../../spec/cgra_cstore_vitra/artifacts/operations.json 2>&1 | %FileCheck %s
// CHECK: Invalid LOAD DFG node
// CHECK-SAME: unsupported explicit-address range/pattern
