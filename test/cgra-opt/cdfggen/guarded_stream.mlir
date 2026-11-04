// RUN: python3 %S/test_guarded_stream.py %cgra-opt %s
module {
 func.func @guarded_stream(%src: memref<64xi16>, %mx: memref<16xi16>, %my: memref<16xi16>, %dst: memref<16xi16>) {
  %s = ADORA.BlockLoad %src [0] : memref<64xi16> -> memref<64xi16> {Id = "src", KernelName = "guarded_stream"}
  %xs = ADORA.BlockLoad %mx [0] : memref<16xi16> -> memref<16xi16> {Id = "x", KernelName = "guarded_stream"}
  %ys = ADORA.BlockLoad %my [0] : memref<16xi16> -> memref<16xi16> {Id = "y", KernelName = "guarded_stream"}
  %out = ADORA.LocalMemAlloc memref<16xi16> {Id = "out", KernelName = "guarded_stream"}
  ADORA.kernel {
   affine.for %i = 0 to 16 {
    %zero = arith.constant 0 : i16
    %eight = arith.constant 8 : i16
    %width = arith.constant 8 : index
    %border = arith.constant -5536 : i16
    %x = affine.load %xs[%i] : memref<16xi16>
    %y = affine.load %ys[%i] : memref<16xi16>
    %xlo = arith.cmpi sge, %x, %zero : i16
    %xhi = arith.cmpi slt, %x, %eight : i16
    %gx = arith.andi %xlo, %xhi : i1
    %ylo = arith.cmpi sge, %y, %zero : i16
    %yhi = arith.cmpi slt, %y, %eight : i16
    %gy = arith.andi %ylo, %yhi : i1
    %v = scf.if %gx -> (i16) {
     %inner = scf.if %gy -> (i16) {
      %ix = arith.index_cast %x : i16 to index
      %iy = arith.index_cast %y : i16 to index
      %row = arith.muli %iy, %width : index
      %address = arith.addi %row, %ix : index
      %pixel = memref.load %s[%address] : memref<64xi16>
      scf.yield %pixel : i16
     } else {
      scf.yield %border : i16
     }
     scf.yield %inner : i16
    } else {
     scf.yield %border : i16
    }
    affine.store %v, %out[%i] : memref<16xi16>
   }
   ADORA.terminator
  } {KernelName = "guarded_stream"}
  ADORA.BlockStore %out, %dst [0] : memref<16xi16> -> memref<16xi16> {Id = "out", KernelName = "guarded_stream"}
  return
 }
}
