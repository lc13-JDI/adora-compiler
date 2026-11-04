// RUN: rm -rf %t && mkdir -p %t
// RUN: python3 %S/test_dynamic_load_loop_mapping.py %cgra-mapper %S/../../spec/cgra_cstore_vitra/artifacts/adg.json %S/../../spec/cgra_cstore_vitra/artifacts/operations.json %S/../../cgra-opt/cdfggen/dynamic_load_loop.mlir %t
