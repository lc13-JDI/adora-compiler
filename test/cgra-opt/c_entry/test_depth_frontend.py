"""Real Clang/cgeist semantic regression, with auditable per-case artifacts.

Run explicitly with --work-dir in a new directory; optional external cgeist is
not assumed to be installed for the ordinary lit suite.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--cgra-opt',type=Path,required=True)
p.add_argument('--cgeist',type=Path,required=True)
p.add_argument('--clang',type=Path,required=True)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--contract',type=Path,required=True)
p.add_argument('--work-dir',type=Path,required=True)
p.add_argument('--driver',type=Path)
p.add_argument('--adg',type=Path)
p.add_argument('--operations',type=Path)
a=p.parse_args()
if a.driver:
 assert all(path is not None and path.is_file() for path in (a.driver,a.adg,a.operations)), 'driver and hardware inputs must exist'
repo=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(repo/'tools/adoracc'))
from c_entry import validate_contract,validate_source_ast,validate_large_source_address,normalize_unsigned_promotions
from adoracc import strip_module_attrs
root=a.work_dir.resolve();root.mkdir(parents=True,exist_ok=False)
source=a.source.read_text();contract=json.loads(a.contract.read_text())
cases=[]
def add(name,text=source,c=contract,good=True): cases.append((name,text,copy.deepcopy(c),good))
add('canonical')
c=copy.deepcopy(contract);c['entry']='different_depth_function'
add('renamed',source.replace('depth_range_overlay','different_depth_function').replace('xx','column').replace('yy','row'),c)
add('reversed_range',source.replace('LO <= v && v <= HI','HI >= v && v >= LO'))
c=copy.deepcopy(contract)
c['objects'][0],c['objects'][1]=c['objects'][1],c['objects'][0]
for i,o in enumerate(c['objects']):o['argument_position']=i
c['host_plan']['bind_arguments']=[0,2,3]
add('parameter_order',source.replace('const uint16_t *src, const int16_t *x','const int16_t *x, const uint16_t *src'),c)
for lo,hi in ((500,4000),(32768,60000),(5000,10000),(501,3999),(1,65535)):
 c=copy.deepcopy(contract);c['defines'].update(LO=lo,HI=hi)
 add(f'range_{lo}_{hi}',c=c)
for name,before,after in [
 ('scatter','dst[i] =','dst[i+1] ='),
 ('feedback','? v : 0','? v : dst[i]'),
 ('extra_load','? v : 0','? src[0] : 0'),
 ('second_writer','? v : 0;','? v : 0; dst[0] = v;'),
 ('missing_negative_guard','xx >= 0 && ',' '),
 ('missing_enable','enabled[i] != 0 && ',' '),
 ('wrong_bound','xx < W','xx <= W'),
 ('wrong_stride','(size_t)yy * (size_t)W','(size_t)yy * (size_t)(W-1)'),
 ('narrow_address','src[(size_t)yy * (size_t)W + (size_t)xx]','src[(unsigned short)((size_t)yy * (size_t)W + (size_t)xx)]'),
 ('volatile','const uint16_t *src','const volatile uint16_t *src'),
 ('atomic','const uint16_t *src','const _Atomic uint16_t *src'),
 ('dynamic_loop','i < T','i < enabled[0]'),
 ('signed_source_address','src[(size_t)yy * (size_t)W + (size_t)xx]','src[(long)yy * (long)W + (long)xx]'),
 ('other_comparison','LO <= v','LO <= xx'),
 ('other_read_comparison','LO <= v','LO <= src[0]'),
 ('nonzero_else','? v : 0','? v : 7'),
 ('only_true_write','dst[i] = (LO <= v && v <= HI) ? v : 0;','if (LO <= v && v <= HI) dst[i] = v;'),
 ('different_else_index','dst[i] = (LO <= v && v <= HI) ? v : 0;','if (LO <= v && v <= HI) dst[i] = v; else dst[0] = 0;'),
 ('explicit_signed_cast','LO <= v && v <= HI','LO <= (int16_t)v && (int16_t)v <= HI')]:
 c=copy.deepcopy(contract)
 if name=='explicit_signed_cast':c['defines'].update(LO=32768,HI=60000)
 assert before in source,name
 add(name,source.replace(before,after),c,False)
for name,changes in [('wrong_type',{'host_dtype':'int16'}),('strided',{'contiguous':False}),('alias',{'may_alias_with':[4]})]:
 c=copy.deepcopy(contract);c['objects'][0].update(changes);add(name,c=c,good=False)
results=[]
for name,text,c,good in cases:
 directory=root/name;directory.mkdir();inp=directory/'input.c';inp.write_text(text)
 cp=directory/'contract.json';cp.write_text(json.dumps(c,indent=2))
 commands=[];error=None
 def run(argv):
  r=subprocess.run([str(x) for x in argv],capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
  number=len(commands);commands.append({'argv':[str(x) for x in argv],'returncode':r.returncode})
  (directory/f'{number}.stdout').write_text(r.stdout);(directory/f'{number}.stderr').write_text(r.stderr)
  if r.returncode:raise ValueError(f'command {number} failed: {r.stderr}')
  return r.stdout
 try:
  validate_contract(c,allow_partial=True,allow_large=True)
  pre=directory/'preprocessed.c'
  run([a.clang,'-E','-x','c','-std=c11','-target','x86_64-unknown-linux-gnu',*[f'-D{k}={v}' for k,v in c['defines'].items()],inp,'-o',pre])
  ast=json.loads(run([a.clang,'-x','c','-std=c11','-Xclang','-ast-dump=json','-fsyntax-only',pre]))
  (directory/'ast.json').write_text(json.dumps(ast))
  validate_source_ast(ast,c,allow_large=True)
  normalized,edits=normalize_unsigned_promotions(ast,pre.read_text())
  normal=directory/'normalized.c';normal.write_text(normalized)
  (directory/'unsigned-promotions.json').write_text(json.dumps(edits,indent=2))
  normalized_ast=json.loads(run([a.clang,'-x','c','-std=c11','-Xclang','-ast-dump=json','-fsyntax-only',normal]))
  assert normalize_unsigned_promotions(normalized_ast,normalized)==(normalized,[]), 'normalization is not stable'
  raw=directory/'raw.mlir'
  run([a.cgeist,normal,'-O2','-S','-std=c11','--target=x86_64-unknown-linux-gnu',f'--function={c["entry"]}','-o',raw])
  bridge=directory/'bridge.mlir';bridge.write_text(strip_module_attrs(raw.read_text()))
  proof=directory/'proof.json'
  run([a.cgra_opt,bridge,f'--adora-lower-c-entry=contract={cp} allow-partial-writes=true large-objects=true descriptor={proof}','--verify-each','-o',directory/'local.mlir'])
  checked=json.loads(proof.read_text())
  validate_large_source_address(ast,checked)
  assert checked['value_mode']=='range' and checked['range_lo']==c['defines']['LO'] and checked['range_hi']==c['defines']['HI'], checked
 except ValueError as exc:error=str(exc)
 (directory/'commands.json').write_text(json.dumps(commands,indent=2))
 passed=(error is None)==good
 if not good and a.driver:
  output=directory/'must-not-publish.py'
  argv=[sys.executable,'-B',str(a.driver.resolve()),str(inp),'--profile','guarded-inplace-large',
        '--contract',str(cp),'--adg-path',str(a.adg.resolve()),'--op-file',str(a.operations.resolve()),
        '--work-dir',str(directory/'driver'),'--output',str(output),'--max-iters','1','--timeout','1',
        '--mapper-wall-timeout','1']
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PATH=os.pathsep.join([
       str(a.clang.resolve().parent),str(a.cgeist.resolve().parent),str(a.cgra_opt.resolve().parent),os.environ['PATH']]))
  r=subprocess.run(argv,capture_output=True,text=True,env=env)
  (directory/'driver.stdout').write_text(r.stdout);(directory/'driver.stderr').write_text(r.stderr)
  (directory/'driver-command.json').write_text(json.dumps({'argv':argv,'returncode':r.returncode},indent=2))
  manifest=directory/'driver/adora-c-entry/provenance.json'
  stages=json.loads(manifest.read_text())['commands'] if manifest.exists() else []
  passed = passed and r.returncode != 0 and not output.exists() and not any(stage['stage']=='mapper' for stage in stages)
 results.append({'case':name,'expected':'accept' if good else 'reject','passed':passed,'error':error})
 print(name,'PASS' if passed else 'FAIL',flush=True)
(root/'results.json').write_text(json.dumps(results,indent=2))
assert all(r['passed'] for r in results),json.dumps([r for r in results if not r['passed']],indent=2)
