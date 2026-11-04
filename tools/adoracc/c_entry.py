"""Opt-in, fail-closed ordinary C entry for the bounded guarded-stream profile.

The C++ bridge owns IR proofs and effect validation. This module owns the C
source boundary, hardware/host contracts, orchestration, and artifact provenance.
"""
from __future__ import annotations

import hashlib
import ast
import json
import keyword
import os
from pathlib import Path
import re
import shutil
import subprocess


_IDENTIFIER = re.compile(r'[A-Za-z_][A-Za-z_0-9]*\Z')
TARGET = 'x86_64-unknown-linux-gnu'


def validate_contract(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError('C entry contract must be a JSON object')
    unknown = set(value) - {'entry', 'defines', 'objects', 'source', 'id'}
    if unknown:
        raise ValueError(f'unsupported contract fields: {sorted(unknown)}')
    entry = value.get('entry')
    if not isinstance(entry, str) or not _IDENTIFIER.fullmatch(entry) or keyword.iskeyword(entry):
        raise ValueError('entry must be a C and Python identifier')
    defines = value.get('defines', {})
    if not isinstance(defines, dict):
        raise ValueError('defines must map macro names to integer constants')
    for name, constant in defines.items():
        if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name) or type(constant) is not int:
            raise ValueError('defines must map macro names to integer constants')
    objects = value.get('objects')
    if not isinstance(objects, list) or not objects:
        raise ValueError('objects must provide every pointer argument')
    ordered = []
    for obj in objects:
        if not isinstance(obj, dict) or set(obj) - {'argument_position', 'name', 'elements',
                                                    'host_dtype', 'access_assertion', 'contiguous'}:
            raise ValueError('unsupported object contract fields')
        if type(obj.get('argument_position')) is not int or obj['argument_position'] < 0:
            raise ValueError('argument_position must be a nonnegative integer')
        if type(obj.get('elements')) is not int or obj['elements'] <= 0:
            raise ValueError('elements must be an explicit positive integer')
        if obj.get('host_dtype') not in ('uint16', 'int16'):
            raise ValueError('host_dtype must be uint16 or int16; int32 truncation is unsupported')
        if obj.get('access_assertion') not in ('read', 'write'):
            raise ValueError('access_assertion must be read or write and agree with IR effects')
        if obj.get('contiguous') is not True:
            raise ValueError('objects must explicitly require contiguous=true')
        ordered.append(dict(obj))
    ordered.sort(key=lambda obj: obj['argument_position'])
    if [obj['argument_position'] for obj in ordered] != list(range(len(ordered))):
        raise ValueError('objects must cover consecutive argument positions exactly once')
    return dict(value, defines=dict(defines), objects=ordered)


def validate_hardware(adg: dict, operations: dict, contract: dict) -> int:
    if not isinstance(adg, dict) or not isinstance(operations, dict):
        raise ValueError('ADG and operations must be JSON objects')
    if adg.get('data_width') != 16:
        raise ValueError('guarded-stream requires ADG data_width=16')
    bank = adg.get('iob_spad_bank_size')
    if type(bank) is not int or bank <= 0:
        raise ValueError('ADG must declare positive iob_spad_bank_size in bytes')
    if any(obj['elements'] * 2 > bank for obj in contract['objects']):
        raise ValueError('object exceeds the declared hardware bank capacity')
    modes = adg.get('iob_mode_names', {})
    if not isinstance(modes, dict) or 'SRAM_MODE' not in modes.values():
        raise ValueError('guarded-stream requires hardware SRAM_MODE for dynamic LOAD')
    ops = operations.get('Operations', [])
    if not isinstance(ops, list) or any(not isinstance(op, dict) for op in ops):
        raise ValueError('Operations must be an array of operation records')
    if not any(op.get('name') == 'LOAD' and op.get('numOperands') == 1 and
               op.get('numRes') == 1 for op in ops):
        raise ValueError('operations must declare a one-address, one-result LOAD')
    return bank


def checked_wrapper(contract: dict, factory_name: str = '_adora_make_checked_entry') -> str:
    """Wrap a successfully verified/emitted function without copying pixel data."""
    contract = validate_contract(contract)
    specs = [(o['elements'], o['host_dtype'], o['access_assertion']) for o in contract['objects']]
    entry = contract['entry']
    return f'''
# Public checked ABI. Validation completes before the generated entry creates a stream.
def {factory_name}(_unchecked):
    async def _checked(runtime, *buffers):
        specs = {specs!r}
        if len(buffers) != len(specs):
            raise ValueError('C entry: incorrect number of buffers')
        for position, (buffer, (elements, dtype, access)) in enumerate(zip(buffers, specs)):
            if not isinstance(buffer, np.ndarray):
                raise ValueError(f'C entry argument {{position}}: expected numpy.ndarray')
            if buffer.shape != (elements,) or buffer.dtype != np.dtype(dtype):
                raise ValueError(f'C entry argument {{position}}: incorrect shape or dtype')
            if not buffer.flags.c_contiguous:
                raise ValueError(f'C entry argument {{position}}: expected contiguous storage')
            if access == 'write' and not buffer.flags.writeable:
                raise ValueError(f'C entry argument {{position}}: output must be writable')
        for left in range(len(buffers)):
            for right in range(left + 1, len(buffers)):
                if np.shares_memory(buffers[left], buffers[right]):
                    raise ValueError('C entry: buffers must not alias')
        transport = tuple(buffer.view(np.int16) for buffer in buffers)
        return await _unchecked(runtime, *transport)
    _checked.__name__ = {entry!r}
    _checked.__doc__ = 'Checked fixed-shape C entry; uint16 transport uses an equal-width view.'
    return _checked
{entry} = {factory_name}({entry})
del {factory_name}
'''


def checked_module(mapper_source: str, contract: dict) -> tuple[str, str]:
    """Isolate emitter globals from arbitrary valid public C entry names.

    Only the selected async definition is renamed. C calls/recursion are
    rejected by the source boundary, so no call sites require rewriting.
    Matching the complete emitted argument list distinguishes an entry named
    aux_stream from the emitter's own aux_stream helper.
    """
    contract = validate_contract(contract)
    tree = ast.parse(mapper_source)
    entry = contract['entry']
    arguments = ['runtime', *(f'arg_{i}' for i in range(len(contract['objects'])))]
    matches = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
               and node.name == entry
               and [arg.arg for arg in node.args.args] == arguments
               and not node.args.posonlyargs and not node.args.kwonlyargs
               and not node.args.vararg and not node.args.kwarg]
    if len(matches) != 1:
        raise ValueError('mapper output must contain exactly one entry with the verified ABI signature')
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.arg):
            used.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            used.add(node.name)
        elif isinstance(node, ast.alias):
            used.add(node.asname or node.name.split('.')[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            used.update(node.names)

    def fresh(prefix):
        index = 0
        while f'{prefix}_{index}' in used:
            index += 1
        name = f'{prefix}_{index}'
        used.add(name)
        return name

    private_entry = fresh('_adora_private_entry')
    matches[0].name = private_entry
    factory = fresh('_adora_private_checked_factory')
    private_contract = dict(contract, entry=private_entry)
    private_source = ast.unparse(tree) + '\n' + checked_wrapper(private_contract, factory)
    compile(private_source, '<adora private implementation>', 'exec')
    # The implementation is embedded so the final .py is independently usable.
    # Its readable companion is retained alongside the unmodified mapper output.
    public_factory = '_adora_export_checked_entry'
    public_source = f'''# Checked public C entry; implementation globals are private.
def {public_factory}():
    namespace = {{'__name__': __name__ + '._adora_private', '__file__': __file__}}
    exec(compile({private_source!r}, __file__, 'exec'), namespace)
    entry = namespace[{private_entry!r}]
    entry.__name__ = {entry!r}
    entry.__qualname__ = {entry!r}
    globals().pop({public_factory!r})
    return entry
globals()[{entry!r}] = {public_factory}()
'''
    return public_source, private_source


def _walk(node):
    yield node
    for child in node.get('inner', []):
        yield from _walk(child)


def validate_source_ast(ast: dict, contract: dict) -> None:
    """Check source properties lost by optimized MLIR, using Clang semantic AST."""
    # Clang JSON desugars scalar typedefs but may spell a pointer's pointee with
    # its typedef name. Resolve those names through actual TypedefDecl nodes.
    # A shadowed/ambiguous name is rejected rather than guessing its binding.
    aliases = {}
    for node in _walk(ast):
        if node.get('kind') == 'TypedefDecl':
            ctype = node.get('type', {})
            spelling = ctype.get('desugaredQualType', ctype.get('qualType', ''))
            annotated = any(child.get('kind', '').endswith('Attr') for child in node.get('inner', []))
            aliases.setdefault(node.get('name'), set()).add((spelling, annotated))

    def source_type(node):
        def expand(spelling, seen=()):
            def replace(match):
                name = match.group(0)
                if name not in aliases:
                    return name
                if name in seen or len(aliases[name]) != 1:
                    raise ValueError(f'ambiguous or recursive source typedef: {name}')
                underlying, annotated = next(iter(aliases[name]))
                if annotated:
                    raise ValueError('source typedef annotations are unsupported')
                return expand(underlying, (*seen, name))
            return re.sub(r'[A-Za-z_][A-Za-z_0-9]*', replace, spelling)
        ctype = node.get('type', {})
        # Expand the written type too, retaining checks on typedef attributes.
        written = expand(ctype.get('qualType', ''))
        return expand(ctype.get('desugaredQualType', written))

    matches = [node for node in _walk(ast) if node.get('kind') == 'FunctionDecl' and
               node.get('name') == contract['entry'] and
               any(child.get('kind') == 'CompoundStmt' for child in node.get('inner', []))]
    if len(matches) != 1:
        raise ValueError('entry must select exactly one C function definition')
    function = matches[0]
    params = [node for node in function.get('inner', []) if node.get('kind') == 'ParmVarDecl']
    if len(params) != len(contract['objects']):
        raise ValueError('object contract must cover every C function argument; runtime sizes unsupported')
    for param, obj in zip(params, contract['objects']):
        spelling = source_type(param)
        # These are parsed AST type tokens, never source-code/kernel matching.
        tokens = spelling.replace('*', ' * ').split()
        if 'volatile' in tokens or '_Atomic' in spelling or 'address_space' in spelling:
            raise ValueError('volatile, atomic and address-space/MMIO pointers are unsupported')
        base = [token for token in tokens if token not in ('const', 'restrict')]
        expected = ['unsigned', 'short', '*'] if obj['host_dtype'] == 'uint16' else ['short', '*']
        if base != expected:
            raise ValueError(f'argument {obj["argument_position"]}: source type {spelling!r} contradicts host dtype or is unsupported')
        if obj['access_assertion'] == 'write' and 'const' in tokens:
            raise ValueError('write assertion contradicts const-qualified C input')
    for node in _walk(function):
        kind = node.get('kind', '')
        spelling = source_type(node)
        if 'volatile' in spelling.split() or '_Atomic' in spelling or 'address_space' in spelling:
            raise ValueError('volatile, atomic and address-space/MMIO accesses are unsupported')
        if kind.endswith('Attr'):
            raise ValueError('source attributes (including strong no-access annotations) are unsupported')
        if kind in {'CallExpr', 'AtomicExpr', 'GCCAsmStmt', 'MSAsmStmt', 'BreakStmt',
                    'ContinueStmt', 'GotoStmt', 'IndirectGotoStmt', 'WhileStmt', 'DoStmt'}:
            raise ValueError(f'unsupported source construct: {kind}')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run_profile(args, require_tool, require_local, strip_module_attrs) -> int:
    """Run the complete opt-in path; every required stage is fatal on failure."""
    if args.input.suffix.lower() != '.c':
        raise ValueError('guarded-stream profile requires ordinary .c input')
    if not args.contract or not args.adg_path or not args.op_file or not args.output:
        raise ValueError('guarded-stream requires --contract, --adg-path, --op-file and --output')
    if args.enable_unroll:
        raise ValueError('guarded-stream does not enable generic unrolling')
    source = args.input.resolve()
    contract_path = args.contract.resolve()
    adg_path = args.adg_path.resolve()
    ops_path = args.op_file.resolve()
    output = args.output.resolve()
    if output.suffix != '.py' or output.exists():
        raise ValueError('guarded-stream --output must be a new .py file')
    contract = validate_contract(json.loads(contract_path.read_text()))
    bank = validate_hardware(json.loads(adg_path.read_text()), json.loads(ops_path.read_text()), contract)
    root = args.work_dir.resolve() / 'adora-c-entry'
    root.mkdir(parents=True, exist_ok=False)
    dirs = {name: root / name for name in ('frontend', 'ir', 'dfg', 'map')}
    for directory in dirs.values():
        directory.mkdir()
    source_copy = dirs['frontend'] / 'input.c'
    shutil.copy2(source, source_copy)
    shutil.copy2(contract_path, root / 'contract.original.json')
    shutil.copy2(adg_path, root / 'adg.json')
    shutil.copy2(ops_path, root / 'operations.json')
    normalized = root / 'contract.json'
    normalized.write_text(json.dumps(contract, indent=2) + '\n')
    tools = {'cgeist': require_tool('cgeist'), 'clang': require_tool('clang'),
             'cgra-opt': require_local('cgra-opt'), 'cgra-mapper': require_local('cgra-mapper')}
    name_file = os.environ.get('GeneralOpNameFile')
    if not name_file:
        name_file = next((str(parent / 'lib/DFG/Documents/GeneralOpName.txt')
                          for parent in Path(__file__).resolve().parents
                          if (parent / 'lib/DFG/Documents/GeneralOpName.txt').is_file()), None)
    if not name_file or not Path(name_file).is_file():
        raise ValueError('set GeneralOpNameFile to the compiler GeneralOpName.txt table')
    shutil.copy2(name_file, root / 'GeneralOpName.txt')
    environment = dict(os.environ, GeneralOpNameFile=str(root / 'GeneralOpName.txt'))
    provenance = {'profile': 'guarded-stream', 'target': TARGET, 'status': 'running',
                  'source': {'path': str(source), 'sha256': _sha256(source)},
                  'tools': {name: {'path': str(Path(path).resolve()), 'sha256': _sha256(Path(path))}
                            for name, path in tools.items()}, 'commands': [],
                  'environment': {'GeneralOpNameFile': environment['GeneralOpNameFile']},
                  'scheduling': 'not required: one fixed kernel, explicitly omitted'}
    manifest = root / 'provenance.json'

    def save():
        manifest.write_text(json.dumps(provenance, indent=2) + '\n')

    def run(argv, name, cwd=root, stdout=None):
        log = root / (name + '.log')
        record = {'stage': name, 'argv': [str(arg) for arg in argv], 'cwd': str(cwd), 'log': str(log)}
        provenance['commands'].append(record)
        save()
        print('+', ' '.join(record['argv']), flush=True)
        with log.open('w') as err:
            if stdout:
                with stdout.open('w') as out:
                    completed = subprocess.run(record['argv'], cwd=cwd, env=environment, stdout=out, stderr=err)
            else:
                completed = subprocess.run(record['argv'], cwd=cwd, env=environment, stdout=err, stderr=subprocess.STDOUT)
        record['returncode'] = completed.returncode
        save()
        if completed.returncode:
            raise ValueError(f'{name} failed (exit {completed.returncode}); see {log}')

    try:
        widths = dirs['frontend'] / 'target_widths.h'
        widths.write_text('_Static_assert(__CHAR_BIT__ == 8 && sizeof(short) == 2 && sizeof(int) == 4, "guarded-stream requires 8-bit bytes, 16-bit short and 32-bit int");\n')
        defines = [f'-D{name}={constant}' for name, constant in sorted(contract['defines'].items())]
        # One preprocessor invocation prevents different Clang versions from
        # selecting different macro branches for the AST and MLIR frontends.
        preprocessed = dirs['frontend'] / 'preprocessed.c'
        run([tools['clang'], '-x', 'c', '-std=c11', '-target', TARGET,
             '-include', widths, *defines, '-E', source], 'preprocess', stdout=preprocessed)
        ast = dirs['frontend'] / 'source.ast.json'
        run([tools['clang'], '-x', 'c', '-std=c11', '-target', TARGET,
             '-Xclang', '-ast-dump=json', '-fsyntax-only', preprocessed],
            'source-ast', stdout=ast)
        validate_source_ast(json.loads(ast.read_text()), contract)
        raw = dirs['frontend'] / 'raw.mlir'
        run([tools['cgeist'], preprocessed, '-O2', '-S', '-std=c11', f'--target={TARGET}',
             f'--function={contract["entry"]}', '--show-ast',
             '-o', raw], 'cgeist')
        raw_text = raw.read_text()
        layout = re.search(r'llvm\.data_layout = "([^"]+)"', raw_text)
        triple = re.search(r'llvm\.target_triple = "([^"]+)"', raw_text)
        if not layout or not triple or triple.group(1) != TARGET:
            raise ValueError('cgeist did not preserve the expected target/layout metadata')
        provenance['frontend_target'] = {'llvm.data_layout': layout.group(1),
                                         'llvm.target_triple': triple.group(1),
                                         'short_bits': 16, 'int_bits': 32, 'char_bits': 8,
                                         'checked_by': str(widths)}
        # Polygeist and ADORA use different LLVM revisions. Only the WORK COPY
        # loses module metadata (not operation attributes); raw retains it all.
        bridge_input = dirs['frontend'] / 'bridge-input.mlir'
        bridge_input.write_text(strip_module_attrs(raw_text))
        bridged = dirs['ir'] / 'bridged.mlir'
        run([tools['cgra-opt'], bridge_input,
             f'--adora-lower-c-entry=contract={normalized} bank-bytes={bank}',
             '--verify-each', '-o', bridged], 'c-entry-bridge')
        final_ir = dirs['ir'] / 'final.mlir'
        run([tools['cgra-opt'], bridged, '--adora-kernel-dfg-gen', '--verify-each',
             '-o', final_ir], 'dfg', cwd=dirs['dfg'])
        graphs = list(dirs['dfg'].glob('*_CDFG.dot'))
        if not graphs or any(not graph.stat().st_size or 'undefined' in graph.read_text() for graph in graphs):
            raise ValueError('DFG generation did not produce valid nonempty graphs')
        raw_python = dirs['map'] / 'unchecked.py'
        run([tools['cgra-mapper'], final_ir, f'--adg={root / "adg.json"}',
             f'--op-file={root / "operations.json"}', '--output-type=pytest',
             f'--output={raw_python}', f'--seed={args.seed}', f'--max-iters={args.max_iters}',
             f'--timeout={args.timeout}'], 'mapper', cwd=dirs['map'])
        configs = list(dirs['map'].glob('*_map_result/config.bit'))
        if not raw_python.is_file() or not configs or any(not config.stat().st_size for config in configs):
            raise ValueError('mapper did not produce Python and nonempty hardware configuration')
        generated, private_source = checked_module(raw_python.read_text(), contract)
        (dirs['map'] / 'generated.py').write_text(private_source)
        compile(generated, str(output), 'exec')
        if _sha256(source) != provenance['source']['sha256']:
            raise ValueError('C source changed during compilation')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as stream:
            stream.write(generated)
        provenance['status'] = 'complete'
        provenance['checked_entry'] = {'path': str(output), 'sha256': _sha256(output), 'name': contract['entry']}
    except Exception as exc:
        provenance['status'] = 'failed'
        provenance['error'] = str(exc)
        raise
    finally:
        provenance['artifacts'] = {str(path.relative_to(root)): _sha256(path)
                                   for path in sorted(root.rglob('*')) if path.is_file() and path != manifest}
        save()
    print(f'[adoracc] Checked C entry: {output}\n[adoracc] Provenance: {manifest}')
    return 0
