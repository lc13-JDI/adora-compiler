// RUN: python3 %S/../../../tools/cstore-handoff/validate_handoff.py --package %S/../../spec/cgra_cstore_vitra
// RUN: python3 %S/../../../tools/cstore-handoff/test_validate_handoff.py
// RUN: env ADORA_ADORACC=%adoracc ADORA_CGRA_MAPPER=%cgra-mapper ADORA_CGRA_ADG=%S/../../spec/cgra_cstore_vitra/artifacts/adg.json ADORA_CGRA_OP_FILE=%S/../../spec/cgra_cstore_vitra/artifacts/operations.json ADORA_CGEIST=/nonexistent ADORA_CONTROL_FLOW_OUT=%t.control-flow %S/../../../experiment/jyhu/control-flow/run.sh if_store
// RUN: awk '/type: IOB/ { found=1 } found && /^operations:/ { print; exit }' %t.control-flow/if_store/mapper.log | %FileCheck %s --check-prefix=IOB
// RUN: sed -n '2p' %t.control-flow/summary.tsv | %FileCheck %s --check-prefix=SUMMARY

// IOB: operations: CSTORE, INPUT, LOAD, OUTPUT, STORE,{{[[:blank:]]*}}{{$}}
// IOB-NOT: CLOAD
// SUMMARY: if_store{{[[:space:]]+}}SKIP(cgeist unavailable){{[[:space:]]+}}PASS{{[[:space:]]+}}PASS{{[[:space:]]+}}PASS{{[[:space:]]+}}FAIL(mapper,rc=1,FOR is not supported!){{[[:space:]]+}}FAIL(mapper,rc=1,FOR is not supported!)
