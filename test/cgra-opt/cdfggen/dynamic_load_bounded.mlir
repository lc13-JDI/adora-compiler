// RUN: python3 %S/test_dynamic_load_bounded.py %cgra-opt %s
module {
  func.func @ebag_plain(%arg0: memref<8xi16>, %arg1: memref<2xi16>, %arg2: memref<2xi16>, %arg3: index) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %result = ADORA.BlockLoad %arg0 [0] : memref<8xi16> -> memref<8xi16>  {Id = "w", KernelName = "ebag_plain"}
    %result_0 = ADORA.BlockLoad %arg1 [0] : memref<2xi16> -> memref<2xi16>  {adora.value_range = [0 : i64, 3 : i64], Id = "ids", KernelName = "ebag_plain"}
    %0 = ADORA.LocalMemAlloc memref<1xi16>  {Id = "out0", KernelName = "ebag_plain"}
    %1 = ADORA.LocalMemAlloc memref<1xi16>  {Id = "out1", KernelName = "ebag_plain"}
    ADORA.kernel {
      affine.for %arg4 = 0 to 1 {
        %2 = affine.load %result_0[0] : memref<2xi16>
        %3 = arith.index_cast %2 : i16 to index
        %4 = arith.muli %3, %c2 : index
        %5 = arith.addi %4, %c1 : index
        %6 = memref.load %result[%4] : memref<8xi16>
        %7 = memref.load %result[%5] : memref<8xi16>
        %8 = affine.load %result_0[1] : memref<2xi16>
        %9 = arith.index_cast %8 : i16 to index
        %10 = arith.muli %9, %c2 : index
        %11 = arith.addi %10, %c1 : index
        %12 = memref.load %result[%10] : memref<8xi16>
        %13 = memref.load %result[%11] : memref<8xi16>
        %14 = arith.addi %6, %12 : i16
        affine.store %14, %0[0] : memref<1xi16>
        %15 = arith.addi %7, %13 : i16
        affine.store %15, %1[0] : memref<1xi16>
      }
      ADORA.terminator
    } {KernelName = "ebag_plain"}
    ADORA.BlockStore %0, %arg2 [0] : memref<1xi16> -> memref<2xi16>  {Id = "out0", KernelName = "ebag_plain"}
    ADORA.BlockStore %1, %arg2 [1] : memref<1xi16> -> memref<2xi16>  {Id = "out1", KernelName = "ebag_plain"}
    return
  }
}
