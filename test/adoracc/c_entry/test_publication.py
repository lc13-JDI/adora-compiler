"""Fault injection at the publication boundary; subprocesses model tool output only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('publication_c_entry',ROOT/'tools/adoracc/c_entry.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Publication(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  def put(name,value):
   p=self.root/name;p.write_text(value);return p
  self.output=self.root/'public.py'
  contract={'entry':'example','objects':[dict(argument_position=i,elements=1,host_dtype='uint16',access_assertion=a,contiguous=True) for i,a in enumerate(('read','write'))]}
  self.args=SimpleNamespace(input=put('input.c','void example(unsigned short *a,unsigned short *b) {}'),contract=put('contract.json',json.dumps(contract)),adg_path=put('adg.json',json.dumps({'data_width':16,'iob_spad_bank_size':8192,'iob_mode_names':{'1':'SRAM_MODE'}})),op_file=put('ops.json',json.dumps({'Operations':[{'name':'LOAD','numOperands':1,'numRes':1}]})),output=self.output,work_dir=self.root/'work',profile='guarded-stream',enable_unroll=False,seed=7,max_iters=1,timeout=1)
  self.tools={name:put(name,'tool') for name in ('clang','cgeist','cgra-opt','cgra-mapper')}
  self.environment=patch.dict(os.environ,{'GeneralOpNameFile':str(put('GeneralOpName.txt','LOAD'))});self.environment.start();self.addCleanup(self.environment.stop)
 def fake_run(self,argv,**kwargs):
  tool=Path(argv[0]).name
  if tool=='clang' and '-E' in argv:kwargs['stdout'].write('source')
  elif tool=='clang':
   ast={'kind':'TranslationUnitDecl','inner':[{'kind':'FunctionDecl','name':'example','inner':[{'kind':'ParmVarDecl','type':{'qualType':'unsigned short *'}} for _ in range(2)]+[{'kind':'CompoundStmt'}]}]}
   kwargs['stdout'].write(json.dumps(ast))
  elif tool=='cgeist':Path(argv[argv.index('-o')+1]).write_text('module attributes {llvm.data_layout = "e", llvm.target_triple = "x86_64-unknown-linux-gnu"} {}')
  elif tool=='cgra-opt':
   Path(argv[argv.index('-o')+1]).write_text('module {}')
   if '--adora-kernel-dfg-gen' in argv:(Path(kwargs['cwd'])/'example_CDFG.dot').write_text('LOAD')
  else:
   Path(next(x.split('=',1)[1] for x in argv if x.startswith('--output='))).write_text('import numpy as np\nasync def example(runtime,arg_0,arg_1): return 17\n')
   directory=Path(kwargs['cwd'])/'example_map_result';directory.mkdir();(directory/'config.bit').write_text('config')
  return SimpleNamespace(returncode=0)
 def run_profile(self):
  with patch.object(m.subprocess,'run',self.fake_run):
   return m.run_profile(self.args,lambda name:str(self.tools[name]),lambda name:str(self.tools[name]),lambda s:s)
 def test_success_hash_and_import(self):
  self.assertEqual(self.run_profile(),0)
  manifest=json.loads((self.root/'work/adora-c-entry/provenance.json').read_text())
  self.assertEqual(manifest['checked_entry']['sha256'],hashlib.sha256(self.output.read_bytes()).hexdigest())
  namespace={'__name__':'test_publication','__file__':str(self.output)};exec(self.output.read_text(),namespace)
  self.assertTrue(callable(namespace['example']))
 def test_final_artifact_hash_failure_publishes_nothing(self):
  original=m._sha256
  def fail(path):
   if path.name=='generated.py':raise OSError('injected artifact hash error')
   return original(path)
  with patch.object(m,'_sha256',fail),self.assertRaises(OSError):self.run_profile()
  self.assertFalse(self.output.exists())
  self.assertEqual(json.loads((self.root/'work/adora-c-entry/provenance.json').read_text())['status'],'failed')
 def test_final_manifest_failure_publishes_nothing(self):
  original=Path.write_text
  def fail(path,text,*args,**kwargs):
   if path.name=='provenance.json' and json.loads(text)['status']=='complete':raise OSError('injected manifest error')
   return original(path,text,*args,**kwargs)
  with patch.object(Path,'write_text',fail),self.assertRaises(OSError):self.run_profile()
  self.assertFalse(self.output.exists())
  self.assertEqual(json.loads((self.root/'work/adora-c-entry/provenance.json').read_text())['status'],'failed')
 def test_existing_output_preserved(self):
  self.output.write_text('existing')
  with self.assertRaises(ValueError):self.run_profile()
  self.assertEqual(self.output.read_text(),'existing')
 def test_concurrent_output_preserved(self):
  original=os.link
  def raced(src,dst,*args,**kwargs):
   self.output.write_text('concurrent')
   return original(src,dst,*args,**kwargs)
  with patch.object(os,'link',raced),self.assertRaises(FileExistsError):self.run_profile()
  self.assertEqual(self.output.read_text(),'concurrent')

if __name__=='__main__':unittest.main()
