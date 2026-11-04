// RUN: rm -rf %t && mkdir -p %t
// RUN: python3 %S/test_dynamic_load.py %cgra-mapper %S/../../spec/cgra_cstore_vitra/artifacts/adg.json %S/../../spec/cgra_cstore_vitra/artifacts/operations.json %s %t
module {
  func.func @ebag_padding(%arg0: memref<8xi16>, %arg1: memref<2xi16>, %arg2: memref<2xi16>, %arg3: index) {
    %c3_i16 = arith.constant 3 : i16
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0_i16 = arith.constant 0 : i16
    %result = ADORA.BlockLoad %arg0 [0] : memref<8xi16> -> memref<8xi16>  {Id = "w", KernelName = "ebag_padding"}
    %result_0 = ADORA.BlockLoad %arg1 [0] : memref<2xi16> -> memref<2xi16>  {adora.value_range = [0 : i64, 3 : i64], Id = "ids", KernelName = "ebag_padding"}
    %0 = ADORA.LocalMemAlloc memref<1xi16>  {Id = "out0", KernelName = "ebag_padding"}
    %1 = ADORA.LocalMemAlloc memref<1xi16>  {Id = "out1", KernelName = "ebag_padding"}
    ADORA.kernel {
      affine.for %arg4 = 0 to 1 {
        %2 = affine.load %result_0[0] : memref<2xi16>
        %3 = arith.index_cast %2 : i16 to index
        %4 = arith.muli %3, %c2 : index
        %5 = arith.addi %4, %c1 : index
        %6 = memref.load %result[%4] : memref<8xi16>
        %7 = memref.load %result[%5] : memref<8xi16>
        %8 = arith.cmpi slt, %2, %c3_i16 : i16
        %9 = arith.select %8, %6, %c0_i16 : i16
        %10 = arith.select %8, %7, %c0_i16 : i16
        %11 = affine.load %result_0[1] : memref<2xi16>
        %12 = arith.index_cast %11 : i16 to index
        %13 = arith.muli %12, %c2 : index
        %14 = arith.addi %13, %c1 : index
        %15 = memref.load %result[%13] : memref<8xi16>
        %16 = memref.load %result[%14] : memref<8xi16>
        %17 = arith.cmpi slt, %11, %c3_i16 : i16
        %18 = arith.select %17, %15, %c0_i16 : i16
        %19 = arith.select %17, %16, %c0_i16 : i16
        %20 = arith.addi %9, %18 : i16
        affine.store %20, %0[0] : memref<1xi16>
        %21 = arith.addi %10, %19 : i16
        affine.store %21, %1[0] : memref<1xi16>
      }
      ADORA.terminator
    } {KernelName = "ebag_padding"}
    ADORA.BlockStore %0, %arg2 [0] : memref<1xi16> -> memref<2xi16>  {Id = "out0", KernelName = "ebag_padding"}
    ADORA.BlockStore %1, %arg2 [1] : memref<1xi16> -> memref<2xi16>  {Id = "out1", KernelName = "ebag_padding"}
    return
  }
}
