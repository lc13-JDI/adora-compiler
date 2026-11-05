// Fixture for test_partial_writes.py; exercised by partial_writes.mlir.
module {
  func.func @transparent_remap(%arg0: memref<?xi16>, %arg1: memref<?xi16>, %arg2: memref<?xi16>, %arg3: memref<?xi16>, %arg4: memref<?xi16>) {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg5 = %c0 to %c8 step %c1 {
      %0 = memref.load %arg1[%arg5] : memref<?xi16>
      %1 = arith.extsi %0 : i16 to i32
      %2 = memref.load %arg2[%arg5] : memref<?xi16>
      %3 = arith.extsi %2 : i16 to i32
      %4 = memref.load %arg3[%arg5] : memref<?xi16>
      %5 = arith.extsi %4 : i16 to i32
      %6 = arith.cmpi ne, %5, %c0_i32 : i32
      %7 = arith.cmpi sge, %1, %c0_i32 : i32
      %8 = arith.cmpi slt, %1, %c3_i32 : i32
      %9 = arith.cmpi sge, %3, %c0_i32 : i32
      %10 = arith.cmpi slt, %3, %c2_i32 : i32
      %11 = arith.andi %9, %10 : i1
      %12 = arith.andi %8, %11 : i1
      %13 = arith.andi %7, %12 : i1
      %14 = arith.andi %6, %13 : i1
      scf.if %14 {
        %15 = arith.muli %3, %c3_i32 : i32
        %16 = arith.addi %15, %1 : i32
        %17 = arith.index_cast %16 : i32 to index
        %18 = memref.load %arg0[%17] : memref<?xi16>
        memref.store %18, %arg4[%arg5] : memref<?xi16>
      }
    }
    return
  }
}
