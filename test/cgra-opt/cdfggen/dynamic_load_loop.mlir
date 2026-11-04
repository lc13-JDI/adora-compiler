// RUN: python3 %S/test_dynamic_load_loop.py %cgra-opt %s
module {
 func.func @sum_l4_n4(%arg0: memref<8xi16>, %arg1: memref<4xi16>, %arg2: memref<2xi16>, %arg3: index) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %w = ADORA.BlockLoad %arg0 [0] : memref<8xi16> -> memref<8xi16> {Id = "w", KernelName = "sum_l4_n4"}
  %ids = ADORA.BlockLoad %arg1 [0] : memref<4xi16> -> memref<4xi16> {Id = "ids", KernelName = "sum_l4_n4", adora.value_range = [0 : i64, 3 : i64]}
  %out0 = ADORA.LocalMemAlloc memref<1xi16> {Id = "out0", KernelName = "sum_l4_n4"}
  %out1 = ADORA.LocalMemAlloc memref<1xi16> {Id = "out1", KernelName = "sum_l4_n4"}
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0 : i16
  ADORA.kernel {
   %r0, %r1 = affine.for %j = 0 to 4 iter_args(%s0 = %zero, %s1 = %zero) -> (i16, i16) {
    %id = affine.load %ids[%j] : memref<4xi16>
    %ix = arith.index_cast %id : i16 to index
    %a = arith.muli %ix, %c2 : index
    %b = arith.addi %a, %c1 : index
    %x = memref.load %w[%a] : memref<8xi16>
    %y = memref.load %w[%b] : memref<8xi16>
    %n0 = arith.addi %s0, %x : i16
    %n1 = arith.addi %s1, %y : i16
    affine.yield %n0, %n1 : i16, i16
   }
   affine.store %r0, %out0[0] : memref<1xi16>
   affine.store %r1, %out1[0] : memref<1xi16>
   ADORA.terminator
  } {KernelName = "sum_l4_n4"}
  ADORA.BlockStore %out0, %arg2 [0] : memref<1xi16> -> memref<2xi16> {Id = "out0", KernelName = "sum_l4_n4"}
  ADORA.BlockStore %out1, %arg2 [1] : memref<1xi16> -> memref<2xi16> {Id = "out1", KernelName = "sum_l4_n4"}
  return
 }
}
