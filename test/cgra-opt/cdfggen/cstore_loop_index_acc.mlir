// RUN: python3 %S/test_cstore_loop_index_acc.py %cgra-opt %cgra-mapper %S/../../spec/cgra_cstore_vitra/artifacts/adg.json %S/../../spec/cgra_cstore_vitra/artifacts/operations.json %S

// Supported loop-index CSTORE lowering must replace the physical affine.for
// node with an ACC whose routed operand 0 is the positive induction step.
// The checker also exercises an unsupported empty loop, existing generic ACC,
// and existing CSTORE/STORE regressions so this focused test documents the
// boundary without widening it.

module {
  func.func @loop_index_a(%value: i16, %enable: i1, %output: memref<4xi16>) {
    ADORA.kernel {
      affine.for %i = 0 to 4 {
        ADORA.cond_store %value, %output[%i] if %enable : memref<4xi16>
      }
      ADORA.terminator
    } {KernelName = "loop_index_a"}
    return
  }
}

// -----

module {
  func.func @loop_index_b(%value: i16, %enable: i1, %output: memref<16xi16>) {
    ADORA.kernel {
      affine.for %i = 3 to 13 step 2 {
        ADORA.cond_store %value, %output[%i] if %enable : memref<16xi16>
      }
      ADORA.terminator
    } {KernelName = "loop_index_b"}
    return
  }
}

// -----

module {
  func.func @loop_index_empty(%value: i16, %enable: i1, %output: memref<4xi16>) {
    ADORA.kernel {
      affine.for %i = 4 to 4 {
        ADORA.cond_store %value, %output[%i] if %enable : memref<4xi16>
      }
      ADORA.terminator
    } {KernelName = "loop_index_empty"}
    return
  }
}
