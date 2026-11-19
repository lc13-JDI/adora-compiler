"""Build a checked canvas module and the read-only wire observer."""
import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def daily_frames(contract):
    d = contract['defines']
    size = d['T']
    if size != 353 or d['W'] != 32 or d['H'] != 8:
        raise ValueError(
            'provide --canvas-frames for geometry outside the built-in 32x8, 353-query regression'
        )
    above = d['HI'] + 1 if d['HI'] < 65535 else 0
    for scenario in ('mixed-old', 'mixed-tail', 'all-old', 'old-regular', 'all-outside'):
        for frame in range(2):
            slot = np.arange(size)
            x = ((slot * 7 + 1) % d['W']).astype(np.int16)
            y = ((slot // 3) % d['H']).astype(np.int16)
            enable = np.full(size, -3, np.int16)
            enable[::9] = 0
            x[::13] = -1
            x[[0, 255, 256, size - 1]] = [0, 31, 31, 0]
            y[[0, 255, 256, size - 1]] = [0, 7, 7, 0]
            enable[[0, 255, 256, size - 1]] = -1
            if scenario == 'mixed-old':
                enable[256:] = 0
            elif scenario == 'all-old':
                enable[:] = 0
            elif scenario == 'old-regular':
                enable[:256] = 0
            elif scenario == 'all-outside':
                x %= d['W']
                enable[:] = -1
            src = np.resize(
                np.roll(
                    np.array([0, d['LO'] - 1, d['LO'], d['HI'], above, 0x8000, 0xFFFF], np.uint16),
                    frame,
                ),
                d['W'] * d['H'],
            )
            if scenario == 'all-outside':
                src[:] = 65535 if d['HI'] < 65535 else 0
            old = np.resize(np.roll(np.array([0, 0x8000, 0xFFFF, 0], np.uint16), frame), size)
            yield src, x, y, enable, old


def oracle(arrays, d):
    src, x, y, enable, old = arrays
    guard = (enable != 0) & (x >= 0) & (y >= 0) & (x < d['W']) & (y < d['H'])
    depth = src[y[guard].astype(np.int64) * d['W'] + x[guard]]
    mask = np.where(old != 0, 255, 0).astype(np.uint16)
    mask[guard] = np.where((depth >= d['LO']) & (depth <= d['HI']), 255, 0)
    return mask, guard


def check_daily_native(source, contract, work):
    """Cross-check every synthetic input against the original two C loops."""
    library = work / 'native-canvas.so'
    defines = [f'-D{k}={v}' for k, v in contract['defines'].items()]
    command = [
        os.environ.get('CC', 'cc'),
        '-O3',
        '-std=c11',
        '-fPIC',
        '-shared',
        *defines,
        str(source),
        '-o',
        str(library),
    ]
    subprocess.run(command, check=True)
    fn = getattr(ctypes.CDLL(str(library)), contract['entry'])
    fn.argtypes = [ctypes.c_void_p] * 6
    fn.restype = None
    hashes = []
    for arrays in daily_frames(contract):
        expected, _ = oracle(arrays, contract['defines'])
        output = np.full(len(expected) + 8, 0x5AA5, np.uint16)
        fn(*(a.ctypes.data for a in (*arrays, output[4:-4])))
        np.testing.assert_array_equal(output[4:-4], expected)
        np.testing.assert_array_equal(output[[0, 1, 2, 3, -4, -3, -2, -1]], 0x5AA5)
        hashes.append(hashlib.sha256(output[4:-4].astype('<u2').tobytes()).hexdigest())
    (work / 'native-daily.json').write_text(
        json.dumps(
            dict(command=command, source_sha256=digest(source), public_output_sha256=hashes),
            indent=2,
        )
        + '\n'
    )


def emitted(folder, adg):
    """Match post-allocation transfers to IOB configuration through bank addresses."""
    tree = ast.parse((folder / 'map/unchecked.py').read_bytes())
    (words,) = [
        ast.literal_eval(stmt.value)
        for stmt in tree.body
        if isinstance(stmt, ast.Assign) and getattr(stmt.targets[0], 'id', '').startswith('cfgbit_')
    ]
    config = dict(zip(words[2::3], [lo | hi << 16 for lo, hi in zip(words[::3], words[1::3])]))
    args = {}
    pending = {}
    for stmt in (s for f in tree.body if isinstance(f, ast.AsyncFunctionDef) for s in f.body):
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        if not isinstance(call.func, ast.Attribute) or call.func.attr != 'append':
            continue
        owner = getattr(call.func.value, 'id', '')
        if owner in ('idata', 'odata'):
            (arg,) = [
                n.id
                for n in ast.walk(call.args[0])
                if isinstance(n, ast.Name) and n.id.startswith('arg_')
            ]
            pending[owner[0]] = int(arg.split('_')[1])
        if owner in ('iptrs', 'optrs'):
            addr, length = map(ast.literal_eval, call.args[0].args)
            assert length == 512
            args[addr] = pending.pop(owner[0])
    mod = {m['id']: m for m in adg['sub_modules']}
    instances = {i['id']: i for i in adg['instances']}
    lines = next((folder / 'map').glob('*_map_result/mapped_dfgio.txt')).read_text().splitlines()
    bank_bytes = adg['iob_spad_bank_size']
    slots = {}
    for line in lines:
        if not line.strip() or line.startswith('#'):
            continue
        tag, name, node, slot, latency = [s.strip() for s in line.split(',')]
        inst = instances[int(node)]
        attrs = mod[inst['module_id']]['attributes']
        base = inst['cfg_blk_index'] << adg['cfg_blk_offset']
        width = adg['cfg_data_width']
        bits = sum(
            config.get(base + j, 0) << (j * width)
            for j in range(1 + max(v[1] for v in attrs['configuration'].values()) // width)
        )
        fields = {}
        for field, field_id in attrs['io_controller_cfg_id'].items():
            _, hi, lo = attrs['configuration'][str(field_id)]
            fields[field] = (bits >> lo) & ((1 << (hi - lo + 1)) - 1)
        select, offset = divmod(fields['BaseAddr'] * 2, bank_bytes)
        bank = adg['iob_to_spad_banks'][slot][select]
        address = bank * bank_bytes + offset
        pos = args[address]
        slots[pos] = dict(
            iob=int(node),
            bank=bank,
            address=address,
            bytes=512,
            tag=tag,
            node=name,
            latency=int(latency),
            fields=fields,
            word_offset=offset // 2,
        )
    assert set(slots) == set(range(len(args)))
    names = (
        ('old', 'mask')
        if len(slots) == 2
        else (
            ('source', 'index', 'enabled', 'old', 'mask')
            if len(slots) == 5
            else ('source', 'enabled', 'old', 'mask')
        )
    )
    assert len(slots) == len(names)
    return dict(
        roles={name: slots[i] for i, name in enumerate(names)},
        config_hex=np.asarray(words, dtype='<u2').tobytes().hex(),
    )


def prepare(paths, work, args):
    if args.canvas_mapped:
        mapped = args.canvas_mapped
    else:
        build = work / 'compile'
        build.mkdir()
        contract = json.loads((ROOT / 'test/adoracc/canvas_mask/contract.json').read_text())
        contract['defines']['T'] = 353
        for obj in contract['objects'][1:]:
            obj['elements'] = 353
        spec = build / 'contract.json'
        spec.write_text(json.dumps(contract))
        mapped = build / 'mapped.py'
        cgeist = args.cgeist or shutil.which('cgeist')
        if not cgeist:
            raise ValueError('provide --cgeist for canvas compilation')
        env = dict(
            os.environ,
            PATH=os.pathsep.join(
                [str(args.build_dir / 'bin'), str(Path(cgeist).parent), os.environ['PATH']]
            ),
            GeneralOpNameFile=str(ROOT / 'lib/DFG/Documents/GeneralOpName.txt'),
        )
        cmd = [
            sys.executable,
            '-B',
            str(ROOT / 'tools/adoracc/adoracc.py'),
            str(HERE / 'depth_canvas_mask.c'),
            '--profile',
            'guarded-canvas-mask',
            '--contract',
            str(spec),
            '--work-dir',
            str(build),
            '--adg-path',
            str(paths['adg']),
            '--op-file',
            str(paths['operations']),
            '--seed',
            '7',
            '--max-iters',
            '30',
            '--timeout',
            '30000',
            '--mapper-wall-timeout',
            '120',
            '-o',
            str(mapped),
        ]
        if args.canvas_variant:
            cmd += ['--canvas-variant', args.canvas_variant]
        (build / 'command.json').write_text(json.dumps(cmd))
        with (build / 'compiler.log').open('w') as log:
            subprocess.run(
                cmd, cwd=build, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
    artifacts = mapped.parent / 'adora-c-entry'
    prov = json.loads((artifacts / 'provenance.json').read_text())
    if prov['status'] != 'complete' or digest(mapped) != prov['checked_entry']['sha256']:
        raise ValueError('incomplete or stale mapped canvas entry')
    if args.canvas_variant and args.canvas_variant != prov['verified_host_descriptor'].get(
        'canvas_variant', 'A'
    ):
        raise ValueError('requested canvas variant differs from checked module')
    if any(digest(paths[k]) != digest(artifacts / (k + '.json')) for k in ('adg', 'operations')):
        raise ValueError('canvas mapping and RTL target identities differ')
    for kind in ('regular', 'old-only'):
        for evidence in [
            artifacts / kind / 'map/unchecked.py',
            *list((artifacts / kind / 'map').glob('*_map_result/mapped_dfgio.txt')),
        ]:
            if digest(evidence) != prov['artifacts'][str(evidence.relative_to(artifacts))]:
                raise ValueError(f'stale mapping evidence for {kind}')
    arch = json.loads(paths['adg'].read_text())
    meta = dict(
        kernels={k: emitted(artifacts / k, arch) for k in ('regular', 'old-only')},
        mapped_sha256=digest(mapped),
        contract=json.loads((artifacts / 'contract.original.json').read_text()),
        cfg_base=json.loads(paths['axil'].read_text())['cfgmem_baseaddr'],
        tiles=arch['cgra_tile_num'],
        banks=max(b for bs in arch['iob_to_spad_banks'].values() for b in bs) + 1,
    )
    meta['wire_observer_sha256'] = digest(HERE / 'wire_monitor.cpp')
    (work / 'bindings.json').write_text(json.dumps(meta, indent=2) + '\n')
    if not args.canvas_frames:
        check_daily_native(artifacts / 'frontend/input.c', meta['contract'], work)
    include = subprocess.check_output(
        ['verilator', '--getenv', 'VERILATOR_ROOT'], text=True
    ).strip()
    subprocess.run(
        [
            os.environ.get('CXX', 'c++'),
            '-shared',
            '-fPIC',
            '-O2',
            '-std=c++17',
            '-I' + str(Path(include) / 'include/vltstd'),
            str(HERE / 'wire_monitor.cpp'),
            '-o',
            str(work / 'observer.so'),
        ],
        check=True,
    )
    return dict(
        ADORA_CANVAS_MAPPED=str(mapped),
        ADORA_CANVAS_FRAMES=str(args.canvas_frames or ''),
        TESTCASE='canvas_mask',
    )
