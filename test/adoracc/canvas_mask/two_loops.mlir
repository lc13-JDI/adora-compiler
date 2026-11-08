// RUN: python3 -B %S/test_proof.py cgra-opt
module attributes {} {
  func.func @depth_canvas_mask(%arg0: memref<?xi16>, %arg1: memref<?xi16>, %arg2: memref<?xi16>, %arg3: memref<?xi16>, %arg4: memref<?xi16>, %arg5: memref<?xi16>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c0_i16 = arith.constant 0 : i16
    %c255_i32 = arith.constant 255 : i32
    %c10000_i32 = arith.constant 10000 : i32
    %c5000_i32 = arith.constant 5000 : i32
    %c32_i64 = arith.constant 32 : i64
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<512xi16>
    scf.for %arg6 = %c0 to %c512 step %c1 {
      %0 = memref.load %arg1[%arg6] : memref<?xi16>
      %1 = arith.extsi %0 : i16 to i32
      %2 = memref.load %arg2[%arg6] : memref<?xi16>
      %3 = arith.extsi %2 : i16 to i32
      %4 = memref.load %arg4[%arg6] : memref<?xi16>
      memref.store %4, %alloca[%arg6] : memref<512xi16>
      %5 = memref.load %arg3[%arg6] : memref<?xi16>
      %6 = arith.extsi %5 : i16 to i32
      %7 = arith.cmpi ne, %6, %c0_i32 : i32
      %8 = arith.cmpi sge, %1, %c0_i32 : i32
      %9 = arith.cmpi slt, %1, %c32_i32 : i32
      %10 = arith.cmpi sge, %3, %c0_i32 : i32
      %11 = arith.cmpi slt, %3, %c8_i32 : i32
      %12 = arith.andi %10, %11 : i1
      %13 = arith.andi %9, %12 : i1
      %14 = arith.andi %8, %13 : i1
      %15 = arith.andi %7, %14 : i1
      scf.if %15 {
        %16 = arith.extsi %2 : i16 to i64
        %17 = arith.muli %16, %c32_i64 : i64
        %18 = arith.extsi %0 : i16 to i64
        %19 = arith.addi %17, %18 : i64
        %20 = arith.index_cast %19 : i64 to index
        %21 = memref.load %arg0[%20] : memref<?xi16>
        %22 = arith.extui %21 : i16 to i32
        %23 = arith.cmpi sge, %22, %c5000_i32 : i32
        %24 = scf.if %23 -> (i32) {
          %26 = arith.cmpi sle, %22, %c10000_i32 : i32
          %27 = arith.select %26, %21, %c0_i16 : i16
          %28 = arith.extui %27 : i16 to i32
          scf.yield %28 : i32
        } else {
          scf.yield %c0_i32 : i32
        }
        %25 = arith.trunci %24 : i32 to i16
        memref.store %25, %alloca[%arg6] : memref<512xi16>
      }
    }
    scf.for %arg6 = %c0 to %c512 step %c1 {
      %0 = memref.load %alloca[%arg6] : memref<512xi16>
      %1 = arith.extui %0 : i16 to i32
      %2 = arith.cmpi ne, %1, %c0_i32 : i32
      %3 = arith.select %2, %c255_i32, %c0_i32 : i32
      %4 = arith.trunci %3 : i32 to i16
      memref.store %4, %arg5[%arg6] : memref<?xi16>
    }
    return
  }
}
