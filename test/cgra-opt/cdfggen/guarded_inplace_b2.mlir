// RUN: python3 %S/test_guarded_inplace.py %cgra-opt %s
// Independently authored R006 B2: initialized old destination, explicit safe
// address LOAD and a genuine CSTORE. This is not copied from C frontend output.
module {
 func.func @b2_inplace(%src: memref<4xi16>, %idx: memref<4xi16>, %enabled: memref<4xi16>, %dst: memref<4xi16>) {
  %s = ADORA.BlockLoad %src [0] : memref<4xi16> -> memref<4xi16> {Id = "src", KernelName = "b2_inplace"}
  %ixs = ADORA.BlockLoad %idx [0] : memref<4xi16> -> memref<4xi16> {Id = "idx", KernelName = "b2_inplace"}
  %ens = ADORA.BlockLoad %enabled [0] : memref<4xi16> -> memref<4xi16> {Id = "enabled", KernelName = "b2_inplace"}
  %old = ADORA.BlockLoad %dst [0] : memref<4xi16> -> memref<4xi16> {Id = "dst", KernelName = "b2_inplace"}
  ADORA.kernel {
   affine.for %i = 0 to 4 {
    %zero = arith.constant 0 : i16
    %n = arith.constant 4 : i16
    %k = affine.load %ixs[%i] : memref<4xi16>
    %e = affine.load %ens[%i] : memref<4xi16>
    %on = arith.cmpi ne, %e, %zero : i16
    %lo = arith.cmpi sge, %k, %zero : i16
    %hi = arith.cmpi slt, %k, %n : i16
    %bounds = arith.andi %lo, %hi : i1
    %read_guard = arith.andi %on, %bounds : i1
    %safe = arith.select %read_guard, %k, %zero : i16
    %address = arith.index_cast %safe : i16 to index
    %value = memref.load %s[%address] : memref<4xi16>
    ADORA.cond_store %value, %old[%i] if %read_guard : memref<4xi16>
   }
   ADORA.terminator
  } {KernelName = "b2_inplace"}
  ADORA.BlockStore %old, %dst [0] : memref<4xi16> -> memref<4xi16> {Id = "dst", KernelName = "b2_inplace"}
  return
 }
}
