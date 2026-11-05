// Fixture for test_partial_writes.py; exercised by partial_writes.mlir.
module {
  func.func @guarded_inplace(%arg0: memref<?xi16>, %arg1: memref<?xi16>, %arg2: memref<?xi16>, %arg3: memref<?xi16>) {
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.load %arg1[0] : memref<?xi16>
    %1 = arith.index_cast %0 : i16 to index
    %2 = arith.extsi %0 : i16 to i32
    %3 = affine.load %arg2[0] : memref<?xi16>
    %4 = arith.extsi %3 : i16 to i32
    %5 = arith.cmpi ne, %4, %c0_i32 : i32
    %6 = arith.cmpi sge, %2, %c0_i32 : i32
    %7 = arith.cmpi slt, %2, %c4_i32 : i32
    %8 = arith.andi %6, %7 : i1
    %9 = arith.andi %5, %8 : i1
    scf.if %9 {
      %10 = affine.load %arg0[symbol(%1)] : memref<?xi16>
      affine.store %10, %arg3[0] : memref<?xi16>
    }
    return
  }
}
