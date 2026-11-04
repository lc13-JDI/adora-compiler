// RUN: rm -rf %t && mkdir -p %t
// RUN: python3 %S/test_dynamic_load_reject.py %cgra-mapper %S/../../spec/cgra_cstore_vitra/artifacts/adg.json %S/../../spec/cgra_cstore_vitra/artifacts/operations.json %S/plain.mlir %t
