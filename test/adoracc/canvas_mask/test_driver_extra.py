"""Checked module/publication tests. Fake tool output is not hardware evidence."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/adoracc'))
import canvas_entry as m


def proof(entry='canvas'):
    return dict(schema=m.SCHEMA, support_version=2,
                canvas_variant='A', packing_layout='indexed', value_logic='canvas', module_identity='a' * 64,
                entry=entry, width=16, height=16, source_elements=256, items=353,
                tile_items=256, range_lo=5000, range_hi=10000, source_argument=0,
                x_argument=1, y_argument=2, enable_argument=3, old_argument=4,
                destination_argument=5, ownership='copy',
                regular_entry=m.ENTRIES['regular'], old_only_entry=m.ENTRIES['old-only'])


def sources():
    return {role: 'import numpy as np\nasync def ' + name + '(runtime, ' +
            ', '.join('arg_' + str(i) for i in range(5 if role == 'regular' else 2)) +
            '):\n    runtime.append(' + repr(role) + ')\n    arg_' +
            ('4' if role == 'regular' else '1') + '[:] = 255\n'
            for role, name in m.ENTRIES.items()}


def imported(descriptor=None):
    generated, _ = m.checked_module(sources(), descriptor or proof())
    namespace = {'__name__': 'standalone_canvas', '__file__': '<generated>'}
    exec(generated, namespace)
    return namespace


class Bindings(unittest.IsolatedAsyncioTestCase):
    async def test_public_builtin_names_remain_callable_and_isolated(self):
        for entry in ('getattr', 'ValueError', 'globals', 'compile', 'exec', 'dict', 'len', 'np'):
            with self.subTest(entry=entry):
                ns = imported(proof(entry))
                source, old, output = np.zeros(256, np.uint16), np.ones(353, np.uint16), np.zeros(353, np.uint16)
                x, y, enabled = (np.zeros(353, np.int16) for _ in range(3))
                enabled[0] = 1
                runtime = []
                await ns[entry](runtime, source, x, y, enabled, old, output)
                self.assertEqual(runtime, ['regular', 'old-only'])
                self.assertTrue(np.all(output == 255))

    async def test_metadata_public_name_is_reserved(self):
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            m.checked_module(sources(), proof('CANVAS_DESCRIPTOR'))

    async def test_stale_mixed_roles_and_foreign_plan_rejected_before_dispatch(self):
        first, other = imported(), imported(dict(proof(), module_identity='b' * 64))
        x, y, enabled = (np.zeros(353, np.int16) for _ in range(3))
        enabled[0] = 1
        source, old, output = np.zeros(256, np.uint16), np.ones(353, np.uint16), np.full(353, 39, np.uint16)
        foreign = other['make_plan'](x, y, enabled)
        with self.assertRaises(ValueError):
            await first['execute_plan']([], foreign, source, old, output)
        foreign.close()
        for replacement in (other['_regular_tile'], first['_old_tile']):
            first['_regular_tile'] = replacement
            plan = first['make_plan'](x, y, enabled)
            runtime = []
            with self.assertRaisesRegex(RuntimeError, 'mixed or stale'):
                await plan.execute(runtime, source, old, output)
            self.assertEqual(runtime, [])
            self.assertTrue(np.all(output == 39))


class Boundaries(unittest.TestCase):
    def test_incomplete_or_wrong_geometry_proof_rejected(self):
        for changes in (dict(ownership='borrow'), dict(tile_items=256.0),
                        dict(width=True), dict(height=17), dict(old_argument=5),
                        dict(range_lo=0), dict(range_hi=65536), dict(entry=None),
                        dict(canvas_variant='D'), dict(packing_layout='permuted'), dict(value_logic='predicate')):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                m.validate_proof(dict(proof(), **changes))
        missing = proof()
        del missing['ownership']
        with self.assertRaises(ValueError):
            m.validate_proof(missing)

    def test_indexed_call_module_cannot_bind_to_same_slot_descriptor(self):
        desc = dict(proof(), canvas_variant='C', packing_layout='same-slot')
        with self.assertRaisesRegex(ValueError, 'ABI signature'):
            m.checked_module(sources(), desc)

    def test_private_argument_order_is_checked_before_binding(self):
        kernels = sources()
        kernels['regular'] = kernels['regular'].replace('arg_1, arg_2', 'arg_2, arg_1')
        with self.assertRaisesRegex(ValueError, 'ABI signature'):
            m.checked_module(kernels, proof())

    def test_cache_exact_inventory_hash_role_key_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'config.bit').write_text('fresh configuration')
            digest = m.common._sha256(root / 'config.bit')
            manifest = dict(key='regular-key', files={'config.bit': digest})
            m.verify_cache(root, manifest, 'regular-key')
            with self.assertRaises(ValueError):
                m.verify_cache(root, manifest, 'old-only-key')
            with self.assertRaises(ValueError):
                m.verify_cache(root, dict(key='regular-key', files={'../config.bit': digest}), 'regular-key')
            (root / 'stale').write_text('stale')
            with self.assertRaises(ValueError):
                m.verify_cache(root, manifest, 'regular-key')
            (root / 'stale').unlink()
            (root / 'alias').symlink_to(root / 'config.bit')
            with self.assertRaises(ValueError):
                m.verify_cache(root, manifest, 'regular-key')

    def test_ast_requires_two_signed_inductions_and_preserves_source_effect_rejections(self):
        contract = dict(entry='canvas', objects=[dict(argument_position=i,
                        elements=256, host_dtype='int16' if i in (1, 2, 3) else 'uint16',
                        access_assertion='write' if i == 5 else 'read', contiguous=True)
                        for i in range(6)])
        def source(induction='int', extra=None):
            params = [dict(kind='ParmVarDecl', type=dict(qualType='short *' if i in (1, 2, 3)
                                                       else 'unsigned short *')) for i in range(6)]
            loops = [dict(kind='ForStmt', inner=[dict(kind='DeclStmt', inner=[
                        dict(kind='VarDecl', type=dict(qualType=induction))])]) for _ in range(2)]
            body = dict(kind='CompoundStmt', inner=loops + ([] if extra is None else [extra]))
            return dict(kind='FunctionDecl', name='canvas', inner=params + [body])
        m.common.validate_source_ast(source(), contract, allow_canvas=True)
        for bad in (source('unsigned int'), source('long'), source(extra=dict(kind='CallExpr')),
                    source(extra=dict(kind='DeclRefExpr', type=dict(qualType='volatile unsigned short'))),
                    source(extra=dict(kind='NoAliasAttr'))):
            with self.assertRaises(ValueError):
                m.common.validate_source_ast(bad, contract, allow_canvas=True)


class Publication(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        def put(name, value):
            path = self.root / name
            path.write_text(value)
            return path
        contract = dict(entry='canvas', objects=[dict(argument_position=i,
                        elements=256 if i == 0 else 353,
                        host_dtype='int16' if i in (1, 2, 3) else 'uint16',
                        access_assertion='write' if i == 5 else 'read', contiguous=True)
                        for i in range(6)],
                        host_plan=dict(strategy='per-query', tile_items=256,
                                       bind_arguments=[1, 2, 3], ownership='copy'))
        self.args = SimpleNamespace(input=put('input.c', 'void canvas() {}'),
                    contract=put('contract.json', json.dumps(contract)),
                    adg_path=put('adg.json', json.dumps(dict(data_width=16, iob_spad_bank_size=8192,
                                                          iob_mode_names={'1': 'SRAM_MODE'}))),
                    op_file=put('operations.json', json.dumps(dict(Operations=[dict(name='LOAD', numOperands=1, numRes=1)]))),
                    output=self.root / 'public.py', work_dir=self.root / 'work',
                    canvas_variant="A", enable_unroll=False, seed=7, max_iters=1, timeout=1,
                    mapper_wall_timeout=2, kernel_cache=None)
        self.tools = {name: put(name, 'mock executable') for name in ('clang', 'cgeist', 'cgra-opt', 'cgra-mapper')}
        environment = patch.dict(os.environ, {'GeneralOpNameFile': str(put('GeneralOpName.txt', 'LOAD'))})
        environment.start()
        self.addCleanup(environment.stop)
        self.omit_old_config = False
        self.graph = 'Input0[opcode = "Input"];\nOutput1[opcode = "Output"];\nInput0 -> Output1[operand = 0];\n'
        self.mapped = '"Input1"[label = "\\N\\nlat=1"];\n"Output2"[label = "\\N\\nlat=2"];\n"Input1"->"Output2"[label = "lat=0\\nop=0"];\n'

    def fake_popen(self, argv, **kwargs):
        tool = Path(argv[0]).name
        if tool == 'clang':
            kwargs['stdout'].write('{}' if '-E' not in argv else 'preprocessed source')
        elif tool == 'cgeist':
            Path(argv[argv.index('-o') + 1]).write_text(
                'module attributes {llvm.data_layout = "e", llvm.target_triple = "x86_64-unknown-linux-gnu"} {}')
        elif tool == 'cgra-opt':
            Path(argv[argv.index('-o') + 1]).write_text('module {}')
            if '--adora-kernel-dfg-gen' in argv:
                (Path(kwargs['cwd']) / 'kernel_CDFG.dot').write_text(self.graph)
            else:
                option = next(a for a in argv if a.startswith('--adora-lower-c-entry='))
                descriptor_path = Path(option.split('descriptor=', 1)[1])
                role = option.split('canvas-kernel=', 1)[1].split()[0]
                descriptor_path.write_text(json.dumps(dict(proof(), selected_kernel=role)))
        else:
            role = Path(kwargs['cwd']).parent.name
            output = Path(next(a.split('=', 1)[1] for a in argv if a.startswith('--output=')))
            output.write_text(sources()[role])
            directory = Path(kwargs['cwd']) / 'kernel_map_result'
            directory.mkdir()
            if not (role == 'old-only' and self.omit_old_config):
                (directory / 'config.bit').write_text('mock configuration')
            (directory / 'before_map_kernel_CDFG.dot').write_text(self.graph)
            (directory / 'mapped_dfg.dot').write_text(self.mapped)
        return SimpleNamespace(returncode=0, wait=lambda **kwargs: 0)

    def run_profile(self):
        # Deliberately isolate publication: these patches do not establish any
        # source/typed proof, hardware mapping or device execution correctness.
        with patch.object(m.subprocess, 'Popen', self.fake_popen), \
             patch.object(m.common, 'validate_source_ast'), \
             patch.object(m.common, 'validate_large_source_address'):
            return m.run_profile(self.args, lambda name: str(self.tools[name]),
                                 lambda name: str(self.tools[name]), lambda value: value)

    def test_complete_artifact_is_importable_and_hash_matches(self):
        self.assertEqual(self.run_profile(), 0)
        provenance = json.loads((self.args.work_dir / 'adora-c-entry/provenance.json').read_text())
        self.assertEqual(provenance['status'], 'complete')
        self.assertEqual(provenance['checked_entry']['sha256'], hashlib.sha256(self.args.output.read_bytes()).hexdigest())
        ns = {'__name__': 'published', '__file__': str(self.args.output)}
        exec(self.args.output.read_text(), ns)
        self.assertTrue(callable(ns['canvas']))

    def test_missing_second_configuration_publishes_nothing(self):
        self.omit_old_config = True
        with self.assertRaises((ValueError, OSError)):
            self.run_profile()
        self.assertFalse(self.args.output.exists())
        provenance = json.loads((self.args.work_dir / 'adora-c-entry/provenance.json').read_text())
        self.assertEqual(provenance['status'], 'failed')

    def test_artifact_hash_failure_publishes_nothing(self):
        original = m.common._sha256
        def fail(path):
            if path.name == 'generated.py':
                raise OSError('injected artifact hash failure')
            return original(path)
        with patch.object(m.common, '_sha256', fail), self.assertRaises(OSError):
            self.run_profile()
        self.assertFalse(self.args.output.exists())

    def test_complete_manifest_failure_publishes_nothing_and_records_failure(self):
        original = Path.write_text
        def fail(path, text, *args, **kwargs):
            if path.name == 'provenance.json' and json.loads(text)['status'] == 'complete':
                raise OSError('injected manifest failure')
            return original(path, text, *args, **kwargs)
        with patch.object(Path, 'write_text', fail), self.assertRaises(OSError):
            self.run_profile()
        self.assertFalse(self.args.output.exists())
        provenance = json.loads((self.args.work_dir / 'adora-c-entry/provenance.json').read_text())
        self.assertEqual(provenance['status'], 'failed')

    def test_racing_output_is_never_overwritten(self):
        original = os.link
        def race(source, destination, *args, **kwargs):
            self.args.output.write_text('concurrent output')
            return original(source, destination, *args, **kwargs)
        with patch.object(os, 'link', race), self.assertRaises(FileExistsError):
            self.run_profile()
        self.assertEqual(self.args.output.read_text(), 'concurrent output')


if __name__ == '__main__':
    unittest.main()
