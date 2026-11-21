// RUN: python3 %S/test_prefix_exit.py %cgra-opt %s
module {
  func.func @paint(%arg0: memref<?xi16>, %arg1: memref<?xi16>, %arg2: memref<?xi16>, %arg3: memref<?xi16>) {
    %true = arith.constant true
    %false = arith.constant false
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = scf.for %arg4 = %c0 to %c8 step %c1 iter_args(%arg5 = %true) -> (i1) {
      %1 = scf.if %arg5 -> (i1) {
        %2 = memref.load %arg1[%arg4] : memref<?xi16>
        %3 = arith.extsi %2 : i16 to i32
        %4 = memref.load %arg2[%arg4] : memref<?xi16>
        %5 = arith.extsi %4 : i16 to i32
        %6 = arith.cmpi sge, %5, %c0_i32 : i32
        scf.if %6 {
          %7 = memref.load %arg2[%arg4] : memref<?xi16>
          %8 = arith.extsi %7 : i16 to i32
          %9 = arith.cmpi ne, %8, %c0_i32 : i32
          %10 = arith.cmpi sge, %3, %c0_i32 : i32
          %11 = arith.cmpi slt, %3, %c8_i32 : i32
          %12 = arith.andi %10, %11 : i1
          %13 = arith.andi %9, %12 : i1
          scf.if %13 {
            %14 = arith.index_cast %2 : i16 to index
            %15 = memref.load %arg0[%14] : memref<?xi16>
            memref.store %15, %arg3[%arg4] : memref<?xi16>
          }
        }
        scf.yield %6 : i1
      } else {
        scf.yield %false : i1
      }
      scf.yield %1 : i1
    }
    return
  }
}
