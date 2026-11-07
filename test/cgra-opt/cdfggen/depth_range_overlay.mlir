// RUN: python3 -B %S/test_depth_range_dfg.py %cgra-opt %s
module {
  func.func @_adora_large_tile(%arg0: memref<256xi16>, %arg1: memref<256xi16>, %arg2: memref<256xi16>, %arg3: memref<256xi16>) attributes {adora.c_entry, llvm.linkage = #llvm.linkage<external>} {
    %c0_i16 = arith.constant 0 : i16
    %c256_i16 = arith.constant 256 : i16
    %result = ADORA.BlockLoad %arg0 [0] : memref<256xi16> -> memref<256xi16>  {Id = "arg0", KernelName = "_adora_large_tile", adora.host_dtype = "uint16"}
    %result_0 = ADORA.BlockLoad %arg1 [0] : memref<256xi16> -> memref<256xi16>  {Id = "arg1", KernelName = "_adora_large_tile", adora.host_dtype = "int16"}
    %result_1 = ADORA.BlockLoad %arg2 [0] : memref<256xi16> -> memref<256xi16>  {Id = "arg2", KernelName = "_adora_large_tile", adora.host_dtype = "int16"}
    %result_2 = ADORA.BlockLoad %arg3 [0] : memref<256xi16> -> memref<256xi16>  {Id = "arg3", KernelName = "_adora_large_tile", adora.host_dtype = "uint16"}
    ADORA.kernel {
      affine.for %arg4 = 0 to 256 {
        %0 = affine.load %result_0[%arg4] : memref<256xi16>
        %1 = affine.load %result_1[%arg4] : memref<256xi16>
        %2 = arith.cmpi ne, %1, %c0_i16 : i16
        %3 = arith.cmpi sge, %0, %c0_i16 : i16
        %4 = arith.cmpi slt, %0, %c256_i16 : i16
        %5 = arith.andi %3, %4 : i1
        %6 = arith.andi %2, %5 : i1
        scf.if %6 {
          %7 = arith.index_cast %0 : i16 to index
          %8 = memref.load %result[%7] : memref<256xi16>
          %c-32768_i16 = arith.constant -32768 : i16
          %9 = arith.addi %8, %c-32768_i16 : i16
          %c0_i16_3 = arith.constant 0 : i16
          %c27232_i16 = arith.constant 27232 : i16
          %10 = arith.cmpi sge, %9, %c0_i16_3 : i16
          %11 = arith.cmpi sle, %9, %c27232_i16 : i16
          %12 = arith.andi %10, %11 : i1
          %13 = scf.if %12 -> (i16) {
            scf.yield %8 : i16
          } else {
            scf.yield %c0_i16 : i16
          }
          memref.store %13, %result_2[%arg4] : memref<256xi16>
        }
      }
      ADORA.terminator
    } {KernelName = "_adora_large_tile"}
    ADORA.BlockStore %result_2, %arg3 [0] : memref<256xi16> -> memref<256xi16>  {Id = "arg3", KernelName = "_adora_large_tile"}
    return
  }
}

