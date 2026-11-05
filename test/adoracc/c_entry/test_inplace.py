"""Inout ABI preserves caller storage and rejects invalid calls before dispatch."""
import asyncio
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(sys.argv[1]).resolve()
sys.argv = sys.argv[:1]
spec = importlib.util.spec_from_file_location('c_entry', ROOT / 'tools/adoracc/c_entry.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


def contract():
    return {'entry': 'overlay', 'objects': [
        {'argument_position': i, 'elements': 4, 'host_dtype': dtype,
         'access_assertion': access, 'contiguous': True}
        for i, (dtype, access) in enumerate([('uint16', 'read'), ('int16', 'read'),
                                            ('uint16', 'inout')])]}


class InplaceABI(unittest.TestCase):
    def setUp(self):
        self.assertIn('allow_partial', inspect.signature(bridge.validate_contract).parameters,
                      'partial writes require an explicit opt-in contract')

    def test_profile_and_hardware_contract(self):
        with self.assertRaises(ValueError):
            bridge.validate_contract(contract())
        bridge.validate_contract(contract(), allow_partial=True)
        adg = {'data_width': 16, 'iob_spad_bank_size': 8192,
               'iob_mode_names': {'1': 'SRAM_MODE', '2': 'COND_LS_MODE'}}
        load = {'name': 'LOAD', 'numOperands': 1, 'numRes': 1}
        store = {'name': 'CSTORE', 'numOperands': 3, 'numRes': 0}
        with self.assertRaises(ValueError):
            bridge.validate_hardware(adg, {'Operations': [load]}, contract())
        self.assertEqual(bridge.validate_hardware(adg, {'Operations': [load, store]}, contract()), 8192)
        with self.assertRaises(ValueError):
            bridge.validate_hardware(dict(adg, iob_mode_names={'1': 'SRAM_MODE'}),
                                     {'Operations': [load, store]}, contract())

    @unittest.skipUnless(shutil.which('clang'), 'semantic source checks need clang')
    def test_pointer_const_does_not_make_destination_elements_const(self):
        with tempfile.TemporaryDirectory(prefix='adora-inout-qualifiers-') as directory:
            path = Path(directory) / 'input.c'
            def source_ast(destination):
                path.write_text('typedef unsigned short *PixelPtr;\n'
                                'void overlay(const unsigned short *src, const short *idx, '
                                + destination + ') { (void)src; (void)idx; (void)dst; }')
                result = subprocess.run(['clang', '-x', 'c', '-std=c11', '-target', bridge.TARGET,
                                         '-Xclang', '-ast-dump=json', '-fsyntax-only', str(path)],
                                        capture_output=True, text=True, check=True)
                return json.loads(result.stdout)
            for access in ('write', 'inout'):
                value = contract()
                value['objects'][-1]['access_assertion'] = access
                for destination in ('unsigned short *const dst', 'unsigned short dst[const 4]',
                                    'const PixelPtr dst'):
                    with self.subTest(access=access, destination=destination):
                        bridge.validate_source_ast(source_ast(destination), value)
                with self.assertRaises(ValueError):
                    bridge.validate_source_ast(source_ast('const unsigned short *dst'), value)

    def entry(self):
        raw = ('import numpy as np\n'
               'async def overlay(runtime, arg_0, arg_1, arg_2):\n'
               '    runtime.append((arg_0, arg_1, arg_2, arg_2.copy()))\n')
        public, _ = bridge.checked_module(raw, contract(), allow_partial=True)
        namespace = {'__name__': 'generated_test', '__file__': '<generated test>'}
        exec(public, namespace)
        return namespace['overlay']

    def test_old_destination_bits_and_storage_reach_generated_call(self):
        src = np.array([0, 32768, 60000, 65535], np.uint16)
        src.flags.writeable = False
        dst = np.array([65535, 12345, 32768, 7], np.uint16)
        before = dst.copy()
        events = []
        asyncio.run(self.entry()(events, src, np.zeros(4, np.int16), dst))
        self.assertEqual(len(events), 1)
        self.assertTrue(np.shares_memory(events[0][2], dst))
        np.testing.assert_array_equal(events[0][3].view(np.uint16), before)
        np.testing.assert_array_equal(dst, before)

    def test_invalid_inout_calls_have_no_dispatch_or_destination_changes(self):
        src = np.arange(4, dtype=np.uint16)
        idx = np.zeros(4, np.int16)
        dst = np.array([8, 9, 32768, 65535], np.uint16)
        readonly = dst.copy()
        readonly.flags.writeable = False
        shared = np.arange(6, dtype=np.uint16)
        endian = '>' if sys.byteorder == 'little' else '<'
        cases = [(src, idx, readonly), (src, idx, dst[:0]), (src[:0], idx, dst),
                 (src, idx, dst.astype(endian + 'u2')),
                 (src, idx, np.zeros(8, np.uint16)[::2]),
                 (shared[:4], idx, shared[2:])]
        entry = self.entry()
        for args in cases:
            before = args[2].copy()
            events = []
            with self.assertRaises(ValueError):
                asyncio.run(entry(events, *args))
            self.assertEqual(events, [])
            np.testing.assert_array_equal(args[2], before)


if __name__ == '__main__':
    unittest.main()
