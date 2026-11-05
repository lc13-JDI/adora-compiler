// RUN: python3 %S/test_inplace_mapping.py %cgra-mapper %S/../../cgra-opt/cdfggen/guarded_inplace_b2.mlir %S/../../spec/cgra_cstore_vitra/artifacts/adg.json %S/../../spec/cgra_cstore_vitra/artifacts/operations.json %t
// Exercise initialized inout allocation, CSTORE routes and the fixed launch
// time of its autonomous index ACC. The independent B2 source is shared with
// the DFG regression; this file is a lit entry, not a second source baseline.
