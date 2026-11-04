// RUN: python3 %S/test_c_entry.py %cgra-opt %s
module {
  func.func @remap_u16(%arg0: memref<?xi16>, %arg1: memref<?xi16>, %arg2: memref<?xi16>, %arg3: memref<?xi16>) {
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_i32 = arith.constant 8 : i32
    %c-5536_i16 = arith.constant -5536 : i16
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg4 = %c0 to %c16 step %c1 {
      %0 = memref.load %arg1[%arg4] : memref<?xi16>
      %1 = arith.extsi %0 : i16 to i32
      %2 = memref.load %arg2[%arg4] : memref<?xi16>
      %3 = arith.extsi %2 : i16 to i32
      %4 = arith.cmpi sge, %1, %c0_i32 : i32
      %5 = arith.cmpi slt, %1, %c8_i32 : i32
      %6 = arith.cmpi sge, %3, %c0_i32 : i32
      %7 = arith.cmpi slt, %3, %c8_i32 : i32
      %8 = arith.andi %6, %7 : i1
      %9 = arith.andi %5, %8 : i1
      %10 = arith.andi %4, %9 : i1
      %11 = scf.if %10 -> (i16) {
        %12 = arith.muli %3, %c8_i32 : i32
        %13 = arith.addi %12, %1 : i32
        %14 = arith.index_cast %13 : i32 to index
        %15 = memref.load %arg0[%14] : memref<?xi16>
        scf.yield %15 : i16
      } else {
        scf.yield %c-5536_i16 : i16
      }
      memref.store %11, %arg3[%arg4] : memref<?xi16>
    }
    return
  }
}
