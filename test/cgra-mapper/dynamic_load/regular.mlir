// RUN: rm -rf %t && mkdir -p %t
// RUN: python3 %S/test_dynamic_load.py %cgra-mapper %S/../../spec/cgra_cstore_vitra/artifacts/adg.json %S/../../spec/cgra_cstore_vitra/artifacts/operations.json %s %t
module {
  func.func @regular_sum(%arg0: memref<8xi16>, %arg1: memref<2xi16>, %arg2: memref<2xi16>, %arg3: index) {
    %result = ADORA.BlockLoad %arg0 [0] : memref<8xi16> -> memref<8xi16>  {Id = "w", KernelName = "regular_sum"}
    %0 = ADORA.LocalMemAlloc memref<1xi16>  {Id = "out0", KernelName = "regular_sum"}
    %1 = ADORA.LocalMemAlloc memref<1xi16>  {Id = "out1", KernelName = "regular_sum"}
    ADORA.kernel {
      affine.for %arg4 = 0 to 1 {
        %2 = affine.load %result[0] : memref<8xi16>
        %3 = affine.load %result[1] : memref<8xi16>
        %4 = affine.load %result[2] : memref<8xi16>
        %5 = affine.load %result[3] : memref<8xi16>
        %6 = arith.addi %2, %4 : i16
        affine.store %6, %0[0] : memref<1xi16>
        %7 = arith.addi %3, %5 : i16
        affine.store %7, %1[0] : memref<1xi16>
      }
      ADORA.terminator
    } {KernelName = "regular_sum"}
    ADORA.BlockStore %0, %arg2 [0] : memref<1xi16> -> memref<2xi16>  {Id = "out0", KernelName = "regular_sum"}
    ADORA.BlockStore %1, %arg2 [1] : memref<1xi16> -> memref<2xi16>  {Id = "out1", KernelName = "regular_sum"}
    return
  }
}
