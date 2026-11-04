// RUN: cstore-dfg-contract-test load-valid %S/../../spec/cgra_cstore_vitra/artifacts/operations.json
// RUN: not cstore-dfg-contract-test load-no-object %S/../../spec/cgra_cstore_vitra/artifacts/operations.json 2>&1 | %FileCheck %s --check-prefix=META
// RUN: not cstore-dfg-contract-test load-no-size %S/../../spec/cgra_cstore_vitra/artifacts/operations.json 2>&1 | %FileCheck %s --check-prefix=META
// RUN: not cstore-dfg-contract-test load-missing-port %S/../../spec/cgra_cstore_vitra/artifacts/operations.json 2>&1 | %FileCheck %s --check-prefix=PORT
// RUN: not cstore-dfg-contract-test load-wrong-port %S/../../spec/cgra_cstore_vitra/artifacts/operations.json 2>&1 | %FileCheck %s --check-prefix=PORT
// RUN: not cstore-dfg-contract-test load-duplicate-port %S/../../spec/cgra_cstore_vitra/artifacts/operations.json 2>&1 | %FileCheck %s --check-prefix=PORT
// META: Invalid LOAD DFG node
// META-SAME: missing object/size
// PORT: Invalid LOAD DFG node
// PORT-SAME: logical operand
