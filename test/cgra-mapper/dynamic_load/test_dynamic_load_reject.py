#!/usr/bin/env python3
import copy
import json
from pathlib import Path
import subprocess
import sys

mapper, adg_path, ops, source, work = [Path(x).resolve() for x in sys.argv[1:]]
work.mkdir(parents=True,exist_ok=True)
text=source.read_text(); adg=json.loads(adg_path.read_text())
cases=[]
cases.append(('missing-domain',text.replace('adora.value_range = [0 : i64, 3 : i64], ',''),adg,'pytest','bounded dynamic LOAD'))
for output in ('c','sdk'):
    cases.append((output+'-unchecked',text,adg,output,'requires pytest output'))
for kind in ('empty','invalid','exhausted','capacity'):
    hardware=copy.deepcopy(adg)
    if kind=='capacity': hardware['iob_spad_bank_size']=8
    else:
        for key in hardware['iob_to_spad_banks']:
            hardware['iob_to_spad_banks'][key]={'empty':[], 'invalid':[-1], 'exhausted':[0]}[kind]
    cases.append((kind+'-bank',text,hardware,'pytest','I/O execution contract'))
# Missing readback must not be accepted merely because inputs are enabled.
cases.append(('missing-output',text.replace('    ADORA.BlockStore %1, %arg2 [1] : memref<1xi16> -> memref<2xi16>  {Id = "out1", KernelName = "ebag_plain"}\n',''),adg,'pytest',None))
results=[]
for name,ir,hardware,output,diagnostic in cases:
    run=work/name;run.mkdir();src=run/'input.mlir';src.write_text(ir)
    hw=run/'adg.json';hw.write_text(json.dumps(hardware));dest=run/'call'
    cmd=[str(mapper),str(src),'--adg='+str(hw),'--op-file='+str(ops),'--output-type='+output,'--output='+str(dest),'--seed=7','--max-iters=30','--timeout=10000','--obj-opt=false']
    p=subprocess.run(cmd,cwd=run,capture_output=True,text=True,timeout=60)
    log=p.stdout+p.stderr;(run/'mapper.log').write_text(log)
    assert p.returncode>0 and not dest.exists(), (name,p.returncode,log[-3000:])
    if diagnostic: assert diagnostic in log, (name,log[-3000:])
    results.append(dict(case=name,rc=p.returncode,argv=cmd))
(work/'results.json').write_text(json.dumps(results,indent=2))
print(f'{len(results)} incomplete/unsupported calls rejected without output')
