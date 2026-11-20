```
      __   __   __           __   __   __           __   __         __          ___  __  
 /\  |  \ /  \ |__)  /\     /  ` / _` |__)  /\     /  ` /  \  |\/| |__) | |    |__  |__) 
/~~\ |__/ \__/ |  \ /~~\    \__, \__> |  \ /~~\    \__, \__/  |  | |    | |___ |___ |  \ 
                                                                                           
```
## ADORA: Adaptive Dataflow Optimization for Reconfigurable Architectures.

An MLIR project for CGRA SoC ([FDRA Repository](https://github.com/MIONkb/FDRA)).
ADORA includes three compilers designed for the CGRA SoC:

- **`tensor-opt`**

·High-level tensor transformation tool for CGRA-based DNN acceleration.  

·Converts commonly used tensor operations to ADORATensor built-in operations (e.g., `linalg.matmul`-> `ADORATensor.Gemm`).  

·Focuses on operator-level dataflow optimizations before hardware mapping.  

- **`cgra-opt`**

·Mid-level automated lowering and transformation framework.  

·Performs affine optimizations for C kernels and tensor operations not directly supported in the ADORATensor dialect.  

·Generates Data-Flow Graphs (DFGs) for subsequent mapping.  

·Can be executed independently of specific hardware information.  

- **`cgra-mapper`**

·Low-level mapper that maps DFGs to the Architecture Description Graph (ADG).  

·Handles hardware-specific scheduling, placement, and resource allocation.  

·Generates RISC-V execution files for the Rocket+CGRA SoC.

·Or generates pytest files for the AXI-CGRA simulation environment.


### Related repositories

External projects that complement this tree:

| Project | Role |
|--------|------|
| [VITRA-CGRA](https://github.com/MIONkb/VITRA-CGRA) | CGRA RTL / hardware generation (Chipyard-based). |
| [CGRA-Cocotb-Sim](https://github.com/theElysia/CGRA-Cocotb-Sim) | Cocotb-based simulation environment for AXI-CGRA. |

#### If you have any issues related to this repository, please don't hesitate to get in touch!

## Directories:
- **`tools`** : Contains the main functions for `tensor-opt`, `cgra-opt` and `cgra-mapper`

- **`include` / `lib`** : C++ headers and source files for `cgra-opt` and `tensor-opt`

- **`mapper`** : C++ source files for `cgra-mapper`

- **`test`** : LLVM **lit** regression suite (`check-adora`): MLIR inputs and FileCheck expectations for `tensor-opt`, `cgra-opt`, `cgra-mapper`, and `adoracc` kernel flows; `test/spec/` holds JSON architecture / operation specs used by mapper tests.

- **`frontend`** : Checkout location for the optional **adora-onnx-mlir** submodule (ONNX → MLIR entry path). Initialize and build it as in [Build § adora-onnx-mlir](#3-adora-onnx-mlir-optional) below.

- **`build_tools`** : Bash scripts for building LLVM and Adora. See [build_tools/C_Compiler_instruction.md](build_tools/C_Compiler_instruction.md) for how to run the C compiler (adoracc + cgra-mapper) end-to-end.

- **`experiment`** : Includes ML benchmarks and C benchmarks (e.g., Polybench). Follow the instructions below to run them. More benchmarks will be added soon.

- **`env.sh`** : Change the environment variables to your own, and source it.
---

# Build

### 1 LLVM-18  (Unskippable)

<!-- ~~LLVM Commit: `26eb4285b56edd8c897642078d91f16ff0fd3472` (Same as [Polygeist GitHub Repository](https://github.com/llvm/Polygeist))~~ -->

LLVM Commit: `b270525f730be6e7196667925f5a9bfa153262e9` (Same as [ONNX-MLIR v5.0.0](https://github.com/onnx/onnx-mlir/tree/v0.5.0.0))

You can download the specified version from the following link:  [LLVM GitHub Repository](https://github.com/llvm/llvm-project/tree/b270525f730be6e7196667925f5a9bfa153262e9)

To install LLVM, follow script:  
```bash
./build_tools/build_llvm.sh
```
### 2 Adora (This project)

Update the LLVM installation path in the following files to your own path:
- `build_adora.sh`
<!-- - `CMakeLists.txt` -->

After making the changes, run `build_adora.sh`.  
**Tip:** Run `build_adora.sh` in small steps (copy one command at a time) so you can catch errors early.

After a successful CMake build, you can run the regression suite from the build directory:

```bash
cmake --build <your-build-dir> --target check-adora
```

The validated loop-bearing conditional-store path is documented in
[`docs/development/cstore-loop-index-e2e.md`](docs/development/cstore-loop-index-e2e.md).

### 3 adora-onnx-mlir (Optional)

A self-hosted **onnx-mlir** fork lives under [`frontend/adora-onnx-mlir`](frontend/adora-onnx-mlir) (git submodule). From the repo root, initialize and update it with:
```bash
git submodule update --init --recursive
```
Then follow the installation script [`build_tools/build_onnxmlir.sh`](build_tools/build_onnxmlir.sh).

If you only need to run small existing models in experiments, or only use C kernels, you can skip adora-onnx-mlir.

### 4 CGRA (Optional)

You can download and install the CGRA from the appropriate repository:  
[CGRA Repository](https://github.com/MIONkb/VITRA-CGRA)

The optional RTL launcher runs generated Python calls against the pinned CSTORE target. See [Optional full RTL simulation](#optional-full-rtl-simulation).

### 5 Other dependencies you may need

#### For AI benchmarks: `torch-mlir`

Download and install `torch-mlir`.  
Our supported version is **`torch-mlir_20250127.357`**. (Corresponding **torchvision** version: `0.1.6.dev0`.)

You can find the official repository here:  
[Torch MLIR GitHub Repository](https://github.com/llvm/torch-mlir)

#### For C benchmarks: Polygeist

Download and install **Polygeist** to run C benchmarks.  
You can find the Polygeist repository here:  
[Polygeist GitHub Repository](https://github.com/llvm/Polygeist)

---

## Guarded writes to the current output slot

The `guarded-inplace` profile accepts a fixed-trip loop with two exclusive
assignments to the same `dst[i]`. This example uses the four-object
[`overlay.json` contract](docs/development/guarded-inplace-c-entry.md)
with initialized `uint16` source and destination arrays and signed 16-bit
index and mode arrays.

```c
void overlay(const unsigned short *src, const short *idx,
             const short *mode, unsigned short *dst) {
  for (int i = 0; i < M; ++i) {
    int j = idx[i];
    if (mode[i] != 0 && j >= 0 && j < N) {
      unsigned short pixel = src[j];
      if (pixel == 0)
        dst[i] = 7;
      else if (mode[i] == 2)
        dst[i] = pixel;
    }
  }
}
```

With the ADORA tools and `cgeist` on `PATH`, compile from the repository root.

```bash
run=$(mktemp -d)
hw=test/spec/cgra_cstore_vitra/artifacts
export GeneralOpNameFile="$PWD/lib/DFG/Documents/GeneralOpName.txt"
build/bin/adoracc.py overlay.c --profile guarded-inplace --contract overlay.json \
  --adg-path "$hw/adg.json" --op-file "$hw/operations.json" \
  --seed 7 --max-iters 1000 --timeout 120000 --work-dir "$run" -o "$run/mapped.py"
```

Call `await mapped.overlay(runtime, src, idx, mode, dst)`. The generated entry
checks all buffer ranges for overlap before creating the device stream. The
device reads a safe source address on every slot and computes both the data
and enable for one conditional store. When neither arm writes, the initialized
destination retains its old value. Two independent conditionals that can both
write in one iteration fail the mutual-exclusion check.

## Large guarded in-place gathers

Compile ordinary C with `adoracc.py --profile guarded-inplace-large` and the
five-object contract. For a proved direct copy, the generated plan packs the
pixel for each valid query into that query's local slot. The device reads the
packed stream in order and conditionally writes the destination. A valid zero
writes zero. An invalid coordinate or disabled query preserves its old output.

The private direct kernel uploads three 512-byte buffers per active tile
(packed pixels, geometric enable, old output) and reads back 512 bytes. The
indexed ABI uploads a fourth 512-byte buffer. Empty tiles skip the call.
On the pinned full CSTORE RTL, the 353-query direct example measured 2,048 B
versus 1,536 B of application H2D per active block, including its 97-item tail.
Configuration traffic is measured separately. Both window and per-tile
preparation use the same public five-array interface.
Nonzero and range predicates retain their indexed implementation. See the
[ordinary C example and prepared-plan API](docs/development/guarded-inplace-large-c-entry.md).

## Optional full RTL simulation

The default build uses LLVM/MLIR and runs the ordinary regression suite with
`cmake --build "$BUILD" --target check-adora`. Full RTL simulation is opt-in.
The commands below use Bash. Set `ROOT` to the checkout, `LLVM` to the LLVM
build containing `lib/cmake/mlir`, and `BUILD` to a new build directory.

```bash
cmake -S "$ROOT" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_BUILD_DIR="$LLVM" -DLLVM_INSTALL_DIR="$LLVM" \
  -DMLIR_DIR="$LLVM/lib/cmake/mlir" -DADORA_ENABLE_RTL_SIM=OFF
cmake --build "$BUILD" --target adoracc.py cgra-opt cgra-mapper tensor-opt -j2
cmake --build "$BUILD" --target check-adora -j2
```

For device execution, activate a Python environment with cocotb 1.9.2,
cocotbext-axi 0.1.28 and NumPy 2.4.4. Put Verilator, GNU Make and a C++17
compiler on PATH. The tested combination uses Python 3.12.13, Verilator 5.044
and GCC 15.2.0 for the model (`CXX` selects the executable). Icarus runs the separate software ownership suite. Set
`CGEIST` to the Polygeist cgeist executable. The ADORA simulator uses the
single `runtime/python/test_runif.py` implementation for both generated Python
and the C++ TCP server.

Set `RTL_ASSETS` to your offline target directory. Place the canonical
`CGRAWithAXI.v` for the CSTORE generator in `$RTL_ASSETS/cstore/`.
The generation entry is `tram.vitra.CStoreVerilogGen`, VITRA commit
`da03f4ab0cf696466147ac9210518e7ead6c9589`. For the older IntVecAdd
target, place the three `circuits/` files named in its manifest in
`$RTL_ASSETS/intvecadd/`. Their immutable source is simulation commit
`c48135347c8ea98b4c5b5da28ecdd16a67d5fef8`. The manifests identify
each input by SHA-256, including the bundled wrappers and register maps.
Availability of authorized copies of the runtime and legacy assets is a
prerequisite for reproducing the local integration. See the
[CSTORE generator handoff](test/spec/cgra_cstore_vitra/HANDOFF.md),
[target definitions](test/spec/rtl_sim) and
[runtime provenance](runtime/python/README.md).

For a hardware handoff directory `CSTORE_GENERATED` containing the canonical
Verilog, the launcher validates the bytes before installing them. The legacy
target can be assembled directly from the three pinned blobs. Use a fresh
`LEGACY_INPUT` directory for the downloads.

```bash
python -B "$ROOT/build_tools/rtl_sim.py" import --target cstore \
  --from-dir "$CSTORE_GENERATED" --assets "$RTL_ASSETS"
mkdir -p "$LEGACY_INPUT"
base=https://raw.githubusercontent.com/lc13-JDI/CGRA-Cocotb-Sim/c48135347c8ea98b4c5b5da28ecdd16a67d5fef8/circuits
for f in CGRAWithAXI.v axilite_spec.json vitra_cgra_adg.json; do
  curl --fail --location "$base/$f" -o "$LEGACY_INPUT/$f" || exit 1
done
python -B "$ROOT/build_tools/rtl_sim.py" import --target intvecadd \
  --from-dir "$LEGACY_INPUT" --assets "$RTL_ASSETS"
```

Set `RTL_WORK` to a new output root. The following commands can run from
outside the checkout. Each run needs a fresh output directory. A verified
model cache can serve multiple runs.

```bash
unset PYTHONPATH
rtl=(python -B "$ROOT/build_tools/rtl_sim.py")
"${rtl[@]}" verify --target cstore --assets "$RTL_ASSETS"
"${rtl[@]}" run --case short-tail --assets "$RTL_ASSETS" --cache "$RTL_WORK/models" \
  --output "$RTL_WORK/tail" --build-dir "$BUILD" --cgeist "$CGEIST"
"${rtl[@]}" run --case canvas-mask --assets "$RTL_ASSETS" --cache "$RTL_WORK/models" \
  --output "$RTL_WORK/canvas" --build-dir "$BUILD" --cgeist "$CGEIST"
python -B "$ROOT/test/rtl/canvas/decode_wires.py" --run "$RTL_WORK/canvas" --tamper-test
for case_name in vector-add tcp-client; do
  "${rtl[@]}" run --case "$case_name" --assets "$RTL_ASSETS" \
    --cache "$RTL_WORK/models" --output "$RTL_WORK/$case_name"
done
```

The canvas case starts from the ordinary two-loop C fixture, maps both the
regular and old-only kernels and runs 353-slot frames on the CSTORE model.
The decoder reads the captured bus handshakes and checks the returned values
and transfer counts. For long canvas runs, `--optimize` builds a distinct model
with optimized hot code and the required VPI probes. Use the tested Verilator
5.044 on PATH when building it. For the layout proof, A/B/C/D switches and reusable Python
plan, see [Checked private canvas-to-mask C entry](docs/development/guarded-canvas-mask.md).

To run those four cases through CMake with artifacts under
`$BUILD/rtl-results/`, enable the optional target in the configured build.

```bash
cmake -S "$ROOT" -B "$BUILD" -DADORA_ENABLE_RTL_SIM=ON \
  -DADORA_RTL_PYTHON="$(command -v python)" -DADORA_RTL_ASSETS="$RTL_ASSETS" \
  -DADORA_RTL_CGEIST="$CGEIST"
cmake --build "$BUILD" --target check-adora-rtl
```

The short-tail case compiles ordinary C and maps a new kernel. The test
compares two frames of 353 host values and 17 guard words per frame after
device readback. The Python and C++ examples each read back a 20-element
vector sum. The TCP server binds loopback and accepts a selected port
(`--port 0` selects a free one). To use a separately built client, start
the server in its own shell with
`"${rtl[@]}" run --case tcp-server --assets "$RTL_ASSETS" --cache "$RTL_WORK/models" --output "$RTL_WORK/server" --port 0`.
Read the loopback endpoint from `$RTL_WORK/server/server.json`. The server
exits after the last connected client disconnects. Use `--case ownership` with the short-tail
arguments for fault injection and the longer device lease regression.
Regenerate pre-ownership large-plan modules to obtain the private-stream
cleanup protocol. For a target mismatch, check the named input and expected
hash first. For execution failures inspect the output directory and the model
build log in the cache.

# Run an Example

## Quick Start — three built-in examples

Three ready-to-run MLIR kernels live in [`experiment/example/`](experiment/example/):

| Kernel | Description |
|--------|-------------|
| `mvt/mvt.mlir`   | Matrix-Vector Transpose (PolyBench) |
| `attn/attn.mlir` | Scaled dot-product attention (Transformer) |
| `ffn/ffn.mlir`   | Feed-forward network: FC1 + ReLU + FC2 (Transformer) |

Run any of them with `adoracc` (replace `/path/to/adora-compiler` with the actual root):

```bash
ADORACC=/path/to/adora-compiler/build/bin/adoracc.py

# MVT
$ADORACC experiment/example/mvt/mvt.mlir --work-dir . -o ./mvt_result.mlir

# Attention
$ADORACC experiment/example/attn/attn.mlir --work-dir . -o ./attn_result.mlir

# FFN
$ADORACC experiment/example/ffn/ffn.mlir --work-dir . -o ./ffn_result.mlir
```

After a successful run (using `mvt` as example), the current directory will contain:

```
./
├── mvt_result.mlir                      ← final scheduled MLIR
└── adora-cc-ir/
    ├── 1_frontend/                      ← cgeist output (.c input only; empty for .mlir)
    ├── 2_kernel-opt/
    │   └── mvt_opt.mlir
    ├── 3_task-schedule/
    │   ├── mvt.pre.mlir                 ← input snapshot to scheduler
    │   ├── mvt.final.mlir               ← scheduler output
    │   └── mvt.token_graph.dot          ← task dependency graph
    ├── temp/
    │   ├── normalize/
    │   │   ├── mvt_normalized.mlir
    │   │   ├── kernel_mvt_0_CDFG.dot
    │   │   └── kernel_mvt_1_CDFG.dot
    │   ├── kernel-extract/
    │   │   └── mvt_kernel.mlir
    │   └── dfg/
    │       ├── kernel_mvt_0_CDFG.dot    ← final DFG (copied from temp/normalize)
    │       └── kernel_mvt_1_CDFG.dot
    └── pipeline.log
```

---

## Full C-to-Executable Flow

You can run the full C compilation flow (C → adoracc → cgra-mapper) by following **[build_tools/C_Compiler_instruction.md](build_tools/C_Compiler_instruction.md)**.

The `experiment` directory contains pre-transformed MLIR files ready to use.

### Step 1: Set Up Environment Variables
Before running an example, you need to update the environment paths in `env.sh`:

```bash
######################
# Set your own environment paths in env.sh:
# - CGRVOPT_PROJECT_PATH: Path to your `cgra-opt` project (e.g., xxxx/cgra-opt)
# - CGRA_ADG_PATH: Path to the hardware information generated by the Chisel hardware generator
# - CHIPYARD_DIR: Path to Chipyard for final RISC-V compilation and linking
#
# Note:
# - If you only want to generate the DFG and do not need to simulate the benchmark,
#   the CHIPYARD_DIR path is not required.
######################
source env.sh  # Update paths in env.sh as described above.
```
### Step 2: Run 'deriche' example:  C -> mlir -> DFG -> mapping result -> executable file

To run 'deriche' example from Polybench benchmark, you can go with:
```bash
cd experiment/Cbenchmarks/Polybench/medley/deriche/deriche_mini

### From C source file to MLIR. This step can be skipped since the kernel is already in MLIR format.
bash scripts/0_compileCtoMLIR.sh

### For hardware-independent compilation
bash scripts/1_kernel_opt.sh
bash scripts/2_kernel_dfgs.sh

### For mapping and riscv link (Chipyard must be installed)
bash scripts/3_kernel_map.sh
bash scripts/4_get_all_asms.sh
bash scripts/5_compile_and_link.sh

### The final linked RISC-V executable file can now be simulated using tools such as VCS or Verilator.
```


## Related publications

```bibtex
@inproceedings{lou2025adora,
  title={Adora Compiler: End-to-End Optimization for High-Efficiency Dataflow Acceleration and Task Pipelining on CGRAs},
  author={Lou, Jiahang and Zhu, Qilong and Dai, Yuan and Zhong, Zewei and Yin, Wenbo and Wang, Lingli},
  booktitle={2025 62nd ACM/IEEE Design Automation Conference (DAC)},
  year={2025},
  organization={IEEE}
}
```
