// RUN: mkdir -p %t
// RUN: cd %t && %cgra-opt %s --adora-kernel-dfg-gen -o first.mlir
// RUN: cd %t && %cgra-opt first.mlir --adora-kernel-dfg-gen -o second.mlir
// RUN: %FileCheck %s --check-prefix=DFG < %t/local_CDFG.dot
// RUN: %FileCheck %s --check-prefix=IR < %t/second.mlir
// RUN: sed 's/if %write :/if %eligible :/' %s > %t/unguarded.mlir
// RUN: cd %t && not %cgra-opt unguarded.mlir --adora-kernel-dfg-gen 2>&1 | %FileCheck %s --check-prefix=REJECT
// RUN: sed 's/%is_special, %replacement,/%is_special, %mode,/' %s > %t/bad-value.mlir
// RUN: cd %t && not %cgra-opt bad-value.mlir --adora-kernel-dfg-gen 2>&1 | %FileCheck %s --check-prefix=REJECT
// DFG-DAG: opcode = "CSTORE"
// DFG-DAG: opcode = "load"
// DFG-DAG: opcode = "SEL"
// IR: %[[IMM:.*]] = arith.constant 25 : i16
// IR: %[[VALUE:[a-zA-Z0-9_]+]] = memref.load
// IR: %[[DATA:[a-zA-Z0-9_]+]] = arith.select %{{.*}}, %[[IMM]], %[[VALUE]] : i16
// IR: ADORA.cond_store %[[DATA]],
// REJECT: error: bounded dynamic LOAD: inout requires one CSTORE
module {
  func.func @local(%src: memref<4xi16>, %idx: memref<4xi16>, %modes: memref<4xi16>, %dst: memref<4xi16>) {
    %s = ADORA.BlockLoad %src [0] : memref<4xi16> -> memref<4xi16> {Id = "src", KernelName = "local"}
    %ids = ADORA.BlockLoad %idx [0] : memref<4xi16> -> memref<4xi16> {Id = "idx", KernelName = "local"}
    %m = ADORA.BlockLoad %modes [0] : memref<4xi16> -> memref<4xi16> {Id = "mode", KernelName = "local"}
    %old = ADORA.BlockLoad %dst [0] : memref<4xi16> -> memref<4xi16> {Id = "dst", KernelName = "local"}
    ADORA.kernel {
      affine.for %i = 0 to 4 {
        %zero = arith.constant 0 : i16
        %n = arith.constant 4 : i16
        %replacement = arith.constant 25 : i16
        %idxval = affine.load %ids[%i] : memref<4xi16>
        %mode = affine.load %m[%i] : memref<4xi16>
        %lo = arith.cmpi sge, %idxval, %zero : i16
        %hi = arith.cmpi slt, %idxval, %n : i16
        %bounds = arith.andi %lo, %hi : i1
        %safe = arith.select %bounds, %idxval, %zero : i16
        %address = arith.index_cast %safe : i16 to index
        %sample = memref.load %s[%address] : memref<4xi16>
        %is_special = arith.cmpi eq, %sample, %n : i16
        %enabled = arith.cmpi ne, %mode, %zero : i16
        %eligible = arith.ori %is_special, %enabled : i1
        %write = arith.andi %bounds, %eligible : i1
        %data = arith.select %is_special, %replacement, %sample : i16
        ADORA.cond_store %data, %old[%i] if %write : memref<4xi16>
      }
      ADORA.terminator
    } {KernelName = "local"}
    ADORA.BlockStore %old, %dst [0] : memref<4xi16> -> memref<4xi16> {Id = "dst", KernelName = "local"}
    return
  }
}
