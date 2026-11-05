// RUN: python3 %S/test_partial_writes.py %cgra-opt %s
module {
  func.func @nonzero_overlay(%arg0: memref<?xi16>, %arg1: memref<?xi16>, %arg2: memref<?xi16>, %arg3: memref<?xi16>) {
    %c6 = arith.constant 6 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg4 = %c0 to %c6 step %c1 {
      %0 = memref.load %arg1[%arg4] : memref<?xi16>
      %1 = arith.extsi %0 : i16 to i32
      %2 = memref.load %arg2[%arg4] : memref<?xi16>
      %3 = arith.extsi %2 : i16 to i32
      %4 = arith.cmpi ne, %3, %c0_i32 : i32
      %5 = arith.cmpi sge, %1, %c0_i32 : i32
      %6 = arith.cmpi slt, %1, %c4_i32 : i32
      %7 = arith.andi %5, %6 : i1
      %8 = arith.andi %4, %7 : i1
      scf.if %8 {
        %9 = arith.index_cast %0 : i16 to index
        %10 = memref.load %arg0[%9] : memref<?xi16>
        %11 = arith.extui %10 : i16 to i32
        %12 = arith.cmpi ne, %11, %c0_i32 : i32
        scf.if %12 {
          memref.store %10, %arg3[%arg4] : memref<?xi16>
        }
      }
    }
    return
  }
}
