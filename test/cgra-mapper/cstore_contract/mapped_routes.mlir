// RUN: rm -rf %t && mkdir -p %t/first %t/second
// RUN: %cgra-mapper --help | %FileCheck %s --check-prefix=HELP
// RUN: cd %t/first && %cgra-mapper --seed=7 --adg=%S/../../spec/cgra_cstore_vitra/artifacts/adg.json --op-file=%S/../../spec/cgra_cstore_vitra/artifacts/operations.json --output-type=pytest --obj-opt=false --max-iters=30 --timeout=30000 %s --output=%t/first/mapped.py > %t/first/mapper.log
// RUN: test -s %t/first/cstore_route_map_result/mapped_routes.tsv
// RUN: %FileCheck %s --check-prefix=SEED --input-file=%t/first/mapper.log
// RUN: test "$(grep -xc 'Random seed: 7' %t/first/mapper.log)" -eq 1
// RUN: awk -F '\t' 'NR == 1 { exit !(NF == 12 && $1 == "edge_id" && $2 == "src_dfg" && $3 == "dst_dfg" && $4 == "dst_operation" && $5 == "logical_operand" && $6 == "dst_adg" && $7 == "dst_physical_input" && $8 == "src_latency" && $9 == "route_latency" && $10 == "rdu_delay" && $11 == "arrival_latency" && $12 == "target_latency") }' %t/first/cstore_route_map_result/mapped_routes.tsv
// RUN: %FileCheck %s --check-prefix=ROUTES --input-file=%t/first/cstore_route_map_result/mapped_routes.tsv
// RUN: awk -F '\t' 'NR > 1 && (NF != 12 || $5 < 0 || $11 != $8 + $9 + $10) { exit 1 } END { exit NR < 2 }' %t/first/cstore_route_map_result/mapped_routes.tsv
// RUN: tail -n +2 %t/first/cstore_route_map_result/mapped_routes.tsv | cut -f1 | sort -c -n
// RUN: cd %t/second && %cgra-mapper --seed=7 --adg=%S/../../spec/cgra_cstore_vitra/artifacts/adg.json --op-file=%S/../../spec/cgra_cstore_vitra/artifacts/operations.json --output-type=pytest --obj-opt=false --max-iters=30 --timeout=30000 %s --output=%t/second/mapped.py > %t/second/mapper.log
// RUN: diff -u %t/first/cstore_route_map_result/mapped_routes.tsv %t/second/cstore_route_map_result/mapped_routes.tsv

// HELP: --seed=<uint>
// SEED: Random seed: 7
// ROUTES: 0{{[[:space:]]+}}3{{[[:space:]]+}}2{{[[:space:]]+}}CSTORE{{[[:space:]]+}}0{{[[:space:]]+}}274{{[[:space:]]+}}0{{[[:space:]]+}}3{{[[:space:]]+}}5{{[[:space:]]+}}4{{[[:space:]]+}}12{{[[:space:]]+}}12
// ROUTES-NEXT: 2{{[[:space:]]+}}4{{[[:space:]]+}}2{{[[:space:]]+}}CSTORE{{[[:space:]]+}}2{{[[:space:]]+}}274{{[[:space:]]+}}4{{[[:space:]]+}}3{{[[:space:]]+}}9{{[[:space:]]+}}0{{[[:space:]]+}}12{{[[:space:]]+}}12
// ROUTES-NEXT: 5{{[[:space:]]+}}6{{[[:space:]]+}}2{{[[:space:]]+}}CSTORE{{[[:space:]]+}}1{{[[:space:]]+}}274{{[[:space:]]+}}2{{[[:space:]]+}}7{{[[:space:]]+}}0{{[[:space:]]+}}5{{[[:space:]]+}}12{{[[:space:]]+}}12
module {
  func.func @cstore_route(%value: i32, %condition: i1, %output: memref<8xi32>) {
    %index = arith.constant 0 : index
    ADORA.kernel {
      ADORA.cond_store %value, %output[%index] if %condition : memref<8xi32>
      ADORA.terminator
    } {KernelName = "cstore_route"}
    return
  }
}
