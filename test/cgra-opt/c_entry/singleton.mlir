// RUN: python3 %S/test_singleton.py %cgra-opt %s
module {
  func.func @remap_u16(%arg0: memref<?xi16>, %arg1: memref<?xi16>, %arg2: memref<?xi16>, %arg3: memref<?xi16>) {
    %c-1_i16 = arith.constant -1 : i16
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.load %arg1[0] : memref<?xi16>
    %1 = arith.extsi %0 : i16 to i32
    %2 = arith.index_cast %0 : i16 to index
    %3 = affine.load %arg2[0] : memref<?xi16>
    %4 = arith.extsi %3 : i16 to i32
    %5 = arith.index_cast %3 : i16 to index
    %6 = arith.cmpi sge, %1, %c0_i32 : i32
    %7 = arith.cmpi slt, %1, %c1_i32 : i32
    %8 = arith.cmpi sge, %4, %c0_i32 : i32
    %9 = arith.cmpi slt, %4, %c1_i32 : i32
    %10 = arith.andi %8, %9 : i1
    %11 = arith.andi %7, %10 : i1
    %12 = arith.andi %6, %11 : i1
    %13 = scf.if %12 -> (i16) {
      %14 = affine.load %arg0[symbol(%5) + symbol(%2)] : memref<?xi16>
      scf.yield %14 : i16
    } else {
      scf.yield %c-1_i16 : i16
    }
    affine.store %13, %arg3[0] : memref<?xi16>
    return
  }
}
