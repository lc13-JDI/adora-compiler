// RUN: python3 -B %S/test_large_entry.py %cgra-opt %s
module attributes {} {
  func.func @large_overlay(%arg0: memref<?xi16>, %arg1: memref<?xi16>, %arg2: memref<?xi16>, %arg3: memref<?xi16>, %arg4: memref<?xi16>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8193_i64 = arith.constant 8193 : i64
    %c1_i32 = arith.constant 1 : i32
    %c8193_i32 = arith.constant 8193 : i32
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg5 = %c0 to %c4 step %c1 {
      %0 = memref.load %arg1[%arg5] : memref<?xi16>
      %1 = arith.extsi %0 : i16 to i32
      %2 = memref.load %arg2[%arg5] : memref<?xi16>
      %3 = arith.extsi %2 : i16 to i32
      %4 = memref.load %arg3[%arg5] : memref<?xi16>
      %5 = arith.extsi %4 : i16 to i32
      %6 = arith.cmpi ne, %5, %c0_i32 : i32
      %7 = arith.cmpi sge, %1, %c0_i32 : i32
      %8 = arith.cmpi sge, %3, %c0_i32 : i32
      %9 = arith.cmpi slt, %1, %c8193_i32 : i32
      %10 = arith.cmpi slt, %3, %c1_i32 : i32
      %11 = arith.andi %9, %10 : i1
      %12 = arith.andi %8, %11 : i1
      %13 = arith.andi %7, %12 : i1
      %14 = arith.andi %6, %13 : i1
      scf.if %14 {
        %15 = arith.extsi %2 : i16 to i64
        %16 = arith.muli %15, %c8193_i64 : i64
        %17 = arith.extsi %0 : i16 to i64
        %18 = arith.addi %16, %17 : i64
        %19 = arith.index_cast %18 : i64 to index
        %20 = memref.load %arg0[%19] : memref<?xi16>
        %21 = arith.extsi %20 : i16 to i32
        %22 = arith.cmpi ne, %21, %c0_i32 : i32
        scf.if %22 {
          memref.store %20, %arg4[%arg5] : memref<?xi16>
        }
      }
    }
    return
  }
}
