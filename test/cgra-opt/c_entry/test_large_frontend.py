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
from c_entry import validate_contract,validate_source_ast,validate_large_source_address
from adoracc import strip_module_attrs
root=a.work_dir.resolve();root.mkdir(parents=True,exist_ok=False)
source=a.source.read_text();contract=json.loads(a.contract.read_text())
cases=[]
def add(name,text=source,c=contract,good=True): cases.append((name,text,copy.deepcopy(c),good))
add('original')
c=copy.deepcopy(contract);c['entry']='unrelated_image_operation'
add('renamed',re.sub(r'\bxx\b','column_value',re.sub(r'\byy\b','row_value',source.replace('large_overlay','unrelated_image_operation'))),c)
add('guard_order',source.replace('enabled[i] != 0 && xx >= 0 && yy >= 0 && xx < W && yy < H','W > xx && yy < H && xx >= 0 && enabled[i] != 0 && yy >= 0'))
c=copy.deepcopy(contract)
c['objects'][0],c['objects'][1]=c['objects'][1],c['objects'][0]
for i,o in enumerate(c['objects']):o['argument_position']=i
c['host_plan']['bind_arguments']=[0,2,3]
add('parameter_order',source.replace('const uint16_t *src, const int16_t *x','const int16_t *x, const uint16_t *src'),c)
for count in (1,255,256,257,353,307200):
 c=copy.deepcopy(contract);c['defines']['T']=count
 for obj in c['objects'][1:]:obj['elements']=count
 add(f'count_{count}',c=c)
for name,before,after in [
 ('scatter','dst[i] = value','dst[i+1] = value'),
 ('feedback','dst[i] = value','dst[i] = value + dst[i]'),
 ('second_writer','dst[i] = value','dst[i] = value; dst[0] = value'),
 ('missing_negative_guard','xx >= 0 && ',' '),
 ('wrong_stride','(size_t)yy * (size_t)W','(size_t)yy * (size_t)(W-1)'),
 ('narrow_address','src[(size_t)yy * (size_t)W + (size_t)xx]','src[(unsigned short)((size_t)yy * (size_t)W + (size_t)xx)]'),
 ('volatile','const uint16_t *src','const volatile uint16_t *src'),
 ('atomic','const uint16_t *src','const _Atomic uint16_t *src'),
 ('no_access','void large_overlay','__attribute__((annotate("no_access"))) void large_overlay'),
 ('dynamic_loop','i < T','i < enabled[0]'),
 ('break','int32_t xx = x[i], yy = y[i];','if (enabled[i] == 0) break; int32_t xx = x[i], yy = y[i];'),
 ('narrow_induction','int32_t i = 0','int16_t i = 0'),
 ('signed_source_address','src[(size_t)yy * (size_t)W + (size_t)xx]','src[(long)yy * (long)W + (long)xx]'),
 ('zero_write_predicate','if (value != 0)','if (value == 0)')]:
 c=copy.deepcopy(contract)
 if name=='zero_write_predicate':c['defines']['VALUE_NONZERO']=1
 add(name,source.replace(before,after),c,False)
c=copy.deepcopy(contract);c['objects'][0]['may_alias_with']=[4];add('alias_contract',c=c,good=False)
c=copy.deepcopy(contract);c['objects'][0]['elements']=16777217;add('resource_bound',c=c,good=False)
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
  raw=directory/'raw.mlir'
  run([a.cgeist,pre,'-O2','-S','-std=c11','--target=x86_64-unknown-linux-gnu',f'--function={c["entry"]}','-o',raw])
  bridge=directory/'bridge.mlir';bridge.write_text(strip_module_attrs(raw.read_text()))
  proof=directory/'proof.json'
  run([a.cgra_opt,bridge,f'--adora-lower-c-entry=contract={cp} allow-partial-writes=true large-objects=true descriptor={proof}','--verify-each','-o',directory/'local.mlir'])
  validate_large_source_address(ast,json.loads(proof.read_text()))
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
