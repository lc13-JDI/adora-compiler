// RUN: %cgra-opt %s -split-input-file -verify-diagnostics

// The accumulator starts at a fixed invocation boundary.
func.func @outside_loop(%negative: i1) {
  // expected-error @+1 {{requires a direct eight-slot kernel loop}}
  %count = ADORA.prefix_count %negative
  return
}

// -----

// A seven-slot traversal must not load an eight-sample ACC configuration.
func.func @wrong_length(%mode: memref<8xi16>) {
  ADORA.kernel {
    affine.for %i = 0 to 7 {
      %m = affine.load %mode[%i] : memref<8xi16>
      %zero = arith.constant 0 : i16
      %neg = arith.cmpi slt, %m, %zero : i16
      // expected-error @+1 {{requires a direct eight-slot kernel loop}}
      %count = ADORA.prefix_count %neg
    }
    ADORA.terminator
  } {KernelName = "wrong_length"}
  return
}
