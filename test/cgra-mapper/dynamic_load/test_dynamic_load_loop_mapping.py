"""Bounded loop mapping: final packets, recurrence sample and field limits."""
import copy,json,os,subprocess,sys
from pathlib import Path
from audit_dynamic_load_loop import inspect
mapper,adg_path,ops,source,work=map(lambda x:Path(x).resolve(),sys.argv[1:]);work.mkdir(parents=True,exist_ok=True)
text=source.read_text();adg=json.loads(adg_path.read_text());results=[]
def run(name,ir,hardware,seed,expected):
 r=work/name;r.mkdir();(r/'input.mlir').write_text(ir);(r/'adg.json').write_text(json.dumps(hardware));output=r/'mapped.py'
 cmd=[str(mapper),str(r/'input.mlir'),'--adg='+str(r/'adg.json'),'--op-file='+str(ops),'--output-type=pytest','--output='+str(output),f'--seed={seed}','--max-iters=30','--timeout=10000','--obj-opt=false']
 p=subprocess.run(cmd,cwd=r,capture_output=True,text=True,timeout=90);log=p.stdout+p.stderr;(r/'mapper.log').write_text(log)
 assert (p.returncode==0)==expected,(name,p.returncode,log[-3000:])
 assert output.exists()==expected
 results.append(dict(case=name,rc=p.returncode,argv=cmd))
 return r,log
for length,seed in [(4,7),(4,19),(1,7),(1,19)]:
 ir=text.replace('memref<4xi16>',f'memref<{length}xi16>').replace('= 0 to 4 ',f'= 0 to {length} ')
 r,log=run(f'l{length}-s{seed}',ir,adg,seed,True)
 (r/'command.json').write_text(json.dumps(dict(rc=0,kernel='sum_l4_n4',length=length,seed=seed)))
 inspect(r)
# The final target uses 12-bit cycle counters and 6-bit latency. Shrink fields
# only in test ADG copies to establish rejection, never a successful RTL target.
for kind in ['cycles','output-latency','acc-latency']:
 hardware=copy.deepcopy(adg)
 for mod in hardware['sub_modules']:
  attrs=mod.get('attributes',{})
  mapping=attrs.get('io_controller_cfg_id',{}) if kind!='acc-latency' else attrs.get('affine_ctrl_reg_cfg_id',{})
  field='Cycles0' if kind=='cycles' else 'Latency'
  if field in mapping:
   entry=attrs['configuration'][str(mapping[field])];entry[1]=entry[2]
 r,log=run(kind,text,hardware,7,False);assert 'I/O execution contract' in log
(work/'results.json').write_text(json.dumps(results,indent=2));print('loop mapper: 4 packet audits, 3 field-limit rejections PASS')
