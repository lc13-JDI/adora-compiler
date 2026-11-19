"""Opt-in contract, cache and actual graph operand validation boundaries."""
from pathlib import Path
import sys,tempfile,unittest
sys.path.insert(0,str(Path(sys.argv.pop(1))/'tools/adoracc'))
import c_entry
import adoracc
from unittest.mock import patch
import canvas_entry as m

class Driver(unittest.TestCase):
 def contract(self):
  return dict(entry='renamed',objects=[dict(argument_position=i,elements=256,host_dtype='int16' if i in (1,2,3) else 'uint16',access_assertion='write' if i==5 else 'read',contiguous=True) for i in range(6)],host_plan=dict(strategy='per-query',tile_items=256,bind_arguments=[1,2,3],ownership='copy'))
 def test_canvas_cli_defaults_to_same_slot_with_indexed_fallback(self):
  for extra, expected in (([], 'C'), (['--canvas-variant', 'A'], 'A')):
   with patch.object(sys, 'argv', ['adoracc.py', 'input.c', '--profile', 'guarded-canvas-mask', *extra]):
    self.assertEqual(adoracc.parse_args().canvas_variant, expected)
 def test_opt_in_six_objects(self):
  c=self.contract();self.assertEqual(len(c_entry.validate_contract(c,allow_canvas=True)['objects']),6)
  with self.assertRaises(ValueError):c_entry.validate_contract(c,allow_large=True)
  for field in ('already_fused','ignore_side_effects'):
   with self.assertRaises(ValueError):c_entry.validate_contract(dict(c,**{field:True}),allow_canvas=True)
 def test_double_immediate_refused(self):
  graph='A0[opcode = "Input"];\nC1[opcode = "CONST", value="0x00FF"];\nC2[opcode = "CONST", value="0x0000"];\nSEL3[opcode = "SEL"];\nA0 -> SEL3[operand = 2];\nC1 -> SEL3[operand = 1];\nC2 -> SEL3[operand = 0];'
  with self.assertRaisesRegex(ValueError,'immediate'):m.verify_graph(graph)
 def test_ports_and_immediate_are_preserved(self):
  graph='Input0[opcode = "Input"];\nCONST1[opcode = "CONST", value="0x00FF"];\nSEL2[opcode = "SEL"];\nInput0 -> SEL2[operand = 0];\nCONST1 -> SEL2[operand = 1];\nInput0 -> SEL2[operand = 2];'
  mapped='"Input1"[label = "\\N\\nlat=3"];\n"SEL3"[label = "\\N\\nlat=5\\nimm=255\\nimmIdx=1"];\n"Input1"->"SEL3"[label = "lat=0\\nop=0"];\n"Input1"->"SEL3"[label = "lat=0\\nop=2"];'
  m.verify_graph(graph,mapped)
  with self.assertRaises(ValueError):m.verify_graph(graph,mapped.replace('imm=255','imm=0'))
  with self.assertRaises(ValueError):m.verify_graph(graph,mapped.replace('op=2','op=1'))
 def test_missing_kernel_cache_component_refused(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'a').write_text('x')
   with self.assertRaises(ValueError):m.verify_cache(p,{'key':'k','files':{'a':'0'*64}},'k')

unittest.main()
