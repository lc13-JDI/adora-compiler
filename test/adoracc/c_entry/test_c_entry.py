"""Checked C ABI regression; compiler integration is exercised by lit separately."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[3]
sys.argv = sys.argv[:1]
MODULE = ROOT / 'tools/adoracc/c_entry.py'


def contract():
    return {'entry': 'renamed_entry', 'defines': {'M': 4}, 'objects': [
        {'argument_position': 0, 'elements': 4, 'host_dtype': 'uint16',
         'access_assertion': 'read', 'contiguous': True},
        {'argument_position': 1, 'elements': 4, 'host_dtype': 'int16',
         'access_assertion': 'read', 'contiguous': True},
        {'argument_position': 2, 'elements': 4, 'host_dtype': 'uint16',
         'access_assertion': 'write', 'contiguous': True}]}


class CheckedEntry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if MODULE.exists():
            spec = importlib.util.spec_from_file_location('c_entry', MODULE)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def setUp(self):
        self.assertTrue(MODULE.exists(), 'opt-in C entry module is missing')

    def test_contract_rejects_invalid_and_missing_fields(self):
        self.module.validate_contract(contract())
        for field, value in [('elements', 0), ('elements', True), ('host_dtype', 'int32'),
                             ('access_assertion', 'readwrite'), ('contiguous', False)]:
            value_contract = contract()
            value_contract['objects'][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.module.validate_contract(value_contract)
        for mutate in [lambda c: c['objects'][0].pop('elements'),
                       lambda c: c['objects'][0].update(argument_position=1),
                       lambda c: c.update(entry='bad;entry'),
                       lambda c: c.update(defines={'M': '-include evil.h'}),
                       lambda c: c.update(no_access=True)]:
            value_contract = contract()
            mutate(value_contract)
            with self.assertRaises(ValueError):
                self.module.validate_contract(value_contract)

    def make_entry(self):
        namespace = {'np': np}
        exec('async def renamed_entry(runtime, *args):\n'
             '    runtime.append(args)\n'
             '    args[2][:] = args[0]\n', namespace)
        exec(self.module.checked_wrapper(contract()), namespace)
        return namespace['renamed_entry']

    def test_uint16_views_preserve_bits_and_writeback(self):
        entry = self.make_entry()
        src = np.array([0, 32768, 60000, 65535], dtype=np.uint16)
        dst = np.zeros(4, dtype=np.uint16)
        runtime = []
        asyncio.run(entry(runtime, src, np.array([-32768, -1, 0, 32767], np.int16), dst))
        np.testing.assert_array_equal(src, dst)
        self.assertEqual(runtime[0][0].dtype, np.dtype('int16'))
        self.assertTrue(np.shares_memory(runtime[0][0], src))

    def test_generated_entry_names_do_not_shadow_helpers_or_builtins(self):
        self.assertTrue(hasattr(self.module, 'checked_module'),
                        'generated entries need isolated implementation globals')
        for name in ['renamed_entry', 'np', 'len', 'range', 'aux_stream',
                     '_adora_make_checked_entry', '_adora_export_checked_entry', 'globals']:
            value_contract = contract()
            value_contract['entry'] = name
            raw = ('import numpy as np\n'
                   'from numpy import ndarray\n'
                   '_adora_private_entry_0 = 17\n'
                   '_adora_private_checked_factory_0 = 23\n'
                   'async def aux_stream(stream, config, *buffers):\n'
                   '    stream.append(("helper", len(buffers)))\n'
                   f'async def {name}(runtime, arg_0: ndarray, arg_1: ndarray, arg_2: ndarray):\n'
                   '    stream = runtime.create_stream()\n'
                   '    assert _adora_private_entry_0 == 17\n'
                   '    assert _adora_private_checked_factory_0 == 23\n'
                   '    assert len(arg_0) == 4\n'
                   '    for index in range(4):\n'
                   '        arg_2[index] = np.int16(arg_0[index])\n'
                   '    await aux_stream(stream, [], arg_0, arg_1, arg_2)\n')
            public, private = self.module.checked_module(raw, value_contract)
            namespace = {'__name__': 'generated_test', '__file__': '<generated test>'}
            exec(public, namespace)

            class Runtime:
                def __init__(self):
                    self.events = []

                def create_stream(self):
                    self.events.append('stream')
                    return self.events

            with self.subTest(entry=name):
                entry = namespace[name]
                self.assertEqual(entry.__name__, name)
                runtime = Runtime()
                src = np.array([0, 32768, 60000, 65535], np.uint16)
                out = np.zeros(4, np.uint16)
                asyncio.run(entry(runtime, src, np.zeros(4, np.int16), out))
                np.testing.assert_array_equal(src, out)
                self.assertEqual(runtime.events, ['stream', ('helper', 3)])
                runtime = Runtime()
                with self.assertRaises(ValueError):
                    asyncio.run(entry(runtime, src.astype(np.int16), np.zeros(4, np.int16), out))
                self.assertEqual(runtime.events, [])

    def test_mapper_entry_signature_must_match_exactly_once(self):
        bad = 'async def renamed_entry(stream, config):\n    pass\n'
        good = 'async def renamed_entry(runtime, arg_0, arg_1, arg_2):\n    pass\n'
        for raw in (bad, good + good):
            with self.assertRaises(ValueError):
                self.module.checked_module(raw, contract())

    def test_rejections_before_runtime(self):
        entry = self.make_entry()
        src = np.arange(4, dtype=np.uint16)
        idx = np.arange(4, dtype=np.int16)
        dst = np.zeros(4, dtype=np.uint16)
        readonly = dst.copy()
        readonly.flags.writeable = False
        cases = [(src[:0], idx, dst), (src.astype(np.int16), idx, dst),
                 (np.zeros(8, dtype=np.uint16)[::2], idx, dst), (src, idx, readonly),
                 (src, idx, src), (src, src.view(np.int16), dst),
                 (src.reshape(2, 2), idx, dst), (list(src), idx, dst)]
        for args in cases:
            runtime = []
            with self.assertRaises(ValueError):
                asyncio.run(entry(runtime, *args))
            self.assertEqual(runtime, [])

    def test_hardware_capacity_and_width(self):
        hardware = {'data_width': 16, 'iob_spad_bank_size': 8192,
                    'iob_mode_names': {'1': 'SRAM_MODE'}}
        ops = {'Operations': [{'name': 'LOAD', 'numOperands': 1, 'numRes': 1}]}
        self.assertEqual(self.module.validate_hardware(hardware, ops, contract()), 8192)
        for key, value in [('data_width', 32), ('iob_spad_bank_size', 4)]:
            bad = dict(hardware, **{key: value})
            with self.assertRaises(ValueError):
                self.module.validate_hardware(bad, ops, contract())
        for bad_adg, bad_ops in [([], ops), (hardware, []),
                                 (dict(hardware, iob_mode_names=[]), ops),
                                 (hardware, {'Operations': ['LOAD']})]:
            with self.assertRaises(ValueError):
                self.module.validate_hardware(bad_adg, bad_ops, contract())

    @unittest.skipUnless(shutil.which('clang'), 'semantic source checks need clang')
    def test_semantic_source_qualifiers_and_types(self):
        source = ('void renamed_entry(const unsigned short *src, const short *idx, '
                  'unsigned short *dst) { for (int i=0;i<4;++i) dst[i]=src[0]; }')
        with tempfile.TemporaryDirectory(prefix='adora-c-ast-') as directory:
            path = Path(directory) / 'input.c'

            def ast(text):
                path.write_text(text)
                result = subprocess.run(['clang', '-x', 'c', '-std=c11', '-target',
                                         self.module.TARGET, '-Xclang', '-ast-dump=json',
                                         '-fsyntax-only', str(path)],
                                        capture_output=True, text=True, check=True)
                return json.loads(result.stdout)

            self.module.validate_source_ast(ast(source), contract())
            aliases = ('typedef unsigned short Pixel; typedef short Coord;\n' +
                       source.replace('unsigned short', 'Pixel').replace('const short', 'const Coord'))
            self.module.validate_source_ast(ast(aliases), contract())
            rejected = [source.replace('const unsigned short *src', 'const volatile unsigned short *src'),
                        source.replace('const short *idx', 'const int *idx'),
                        source.replace('unsigned short *dst', 'const unsigned short *dst').replace('dst[i]=src[0]', '(void)src[0]'),
                        '__attribute__((annotate("no_speculative_access"))) ' + source,
                        'typedef volatile unsigned short Pixel;\n' + source.replace('src[0]', '((Pixel *)src)[0]'),
                        source.replace('dst[i]=src[0]', '{ if(i) break; dst[i]=src[0]; }')]
            for text in rejected:
                with self.subTest(source=text), self.assertRaises(ValueError):
                    self.module.validate_source_ast(ast(text), contract())


if __name__ == '__main__':
    unittest.main()
