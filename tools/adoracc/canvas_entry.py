"""Opt-in compilation of proven private canvas producers and mask consumers.

All semantic decisions originate in the typed C-entry pass. This driver owns
artifact validation, the two local mappings, and atomic checked publication.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile

import c_entry as common

SCHEMA = 'adora.guarded-canvas-mask.v1'
ENTRIES = {'regular': '_adora_canvas_regular_tile', 'old-only': '_adora_canvas_old_tile'}


def local_contract(role):
    types = ('uint16', 'int16', 'int16', 'uint16', 'uint16') if role == 'regular' else ('uint16', 'uint16')
    return dict(entry=ENTRIES[role], objects=[dict(argument_position=i, elements=256,
                host_dtype=dtype, access_assertion='write' if i == len(types)-1 else 'read',
                contiguous=True) for i, dtype in enumerate(types)])


def verify_graph(text, mapped=None):
    """Fail closed at the actual CDFG/mapper operand and immediate boundary.

    DFGIR::parseDFGJFromMLIRCDFG maps CDFG id N to mapper id N+1. Only
    inserted single-input PASS nodes may intervene in routed value edges.
    Commutative operations may exchange ports; SELECT/comparisons may not.
    """
    nodes = {}
    for name, attributes in re.findall(r'(?m)^([A-Za-z_]+\d+)\s*\[([^\n]*)\];', text):
        values = dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', attributes))
        if 'opcode' in values:
            nodes[name] = values
    if not nodes:
        raise ValueError('canvas DFG has no typed nodes')
    edges = {}
    for source, target, port in re.findall(r'(?m)^(\w+)\s*->\s*(\w+)\[.*?operand\s*=\s*(\d+)', text):
        if source not in nodes or target not in nodes:
            raise ValueError('canvas DFG has an unknown endpoint')
        port = int(port)
        if port in edges.setdefault(target, {}):
            raise ValueError('canvas DFG has duplicate operand ports')
        edges[target][port] = source
    constants = {n: int(a['value'], 0) & 65535 for n, a in nodes.items() if a['opcode'].upper() == 'CONST'}
    for name, inputs in edges.items():
        if sum(source in constants for source in inputs.values()) > 1:
            raise ValueError('canvas DFG exceeds one immediate per processing element: '+name)
    result = dict(nodes=len(nodes), edges=sum(map(len, edges.values())), single_immediate=True)
    if mapped is None:
        return result
    labels = dict(re.findall(r'(?m)^"(\w+)"\[label\s*=\s*"([^\n]*)"\];', mapped))
    inputs = {}
    for source, target, port in re.findall(r'(?m)^"(\w+)"->"(\w+)"\[label\s*=\s*".*?\\nop=(\d+)', mapped):
        port = int(port)
        if port in inputs.setdefault(target, {}):
            raise ValueError('mapped canvas DFG has duplicate operand ports')
        inputs[target][port] = ('node', source)
    for name, label in labels.items():
        imm = re.search(r'\\nimm=(\d+)\\nimmIdx=(\d+)', label)
        if imm:
            value, port = map(int, imm.groups())
            if port in inputs.setdefault(name, {}):
                raise ValueError('mapped canvas immediate conflicts with routed operand')
            inputs[name][port] = ('const', value & 65535)
    def renamed(name):
        match = re.fullmatch(r'([A-Za-z_]+)(\d+)', name)
        return match[1]+str(int(match[2])+1)
    original_names = {renamed(n) for n in nodes if n not in constants}
    def resolve(value, seen=()):
        if value[0] == 'const' or value[1] in original_names:
            return value
        name = value[1]
        if name in seen or not name.lower().startswith('pass') or set(inputs.get(name, {})) != {0}:
            raise ValueError('unproved mapper-inserted value node: '+name)
        return resolve(inputs[name][0], (*seen, name))
    commutative = {'ADD', 'MUL', 'AND', 'OR', 'XOR'}
    for name, attributes in nodes.items():
        if name in constants:
            continue
        target = renamed(name)
        if target not in labels:
            raise ValueError('mapped canvas node missing: '+target)
        expected = {p: ('const', constants[s]) if s in constants else ('node', renamed(s))
                    for p, s in edges.get(name, {}).items()}
        actual = {p: resolve(v) for p, v in inputs.get(target, {}).items()}
        same = (sorted(expected.values()) == sorted(actual.values()) if attributes['opcode'].upper() in commutative
                else expected == actual)
        if not same:
            raise ValueError(f'mapped canvas operand/constant mismatch at {target}: {expected} != {actual}')
    for name in set(labels)-original_names:
        resolve(('node', name))
    result['mapped_operands_verified'] = True
    return result


def verify_cache(directory, manifest, key):
    if not isinstance(manifest, dict) or manifest.get('key') != key or directory.is_symlink():
        raise ValueError('canvas cache identity mismatch')
    files = manifest.get('files')
    paths = list(directory.rglob('*'))
    if (not isinstance(files, dict) or not files or any(p.is_symlink() for p in paths) or
            any(not isinstance(n, str) or Path(n).is_absolute() or '..' in Path(n).parts or str(Path(n)) != n for n in files) or
            set(files) != {str(p.relative_to(directory)) for p in paths if p.is_file()} or
            any(common._sha256(directory/n) != digest for n, digest in files.items())):
        raise ValueError('canvas cache inventory/hash mismatch')


def validate_proof(proof):
    if not isinstance(proof, dict) or proof.get('schema') != SCHEMA:
        raise ValueError('canvas entry: incompatible compiler proof')
    if (proof.get('regular_entry') != ENTRIES['regular'] or proof.get('old_only_entry') != ENTRIES['old-only'] or
            type(proof.get('range_lo')) is not int or type(proof.get('range_hi')) is not int or
            not 0 < proof['range_lo'] <= proof['range_hi'] <= 65535):
        raise ValueError('canvas entry: incomplete kernel/range proof')
    roles = [proof.get(k) for k in ('source_argument', 'x_argument', 'y_argument', 'enable_argument', 'old_argument', 'destination_argument')]
    if any(type(r) is not int for r in roles) or sorted(roles) != list(range(6)):
        raise ValueError('canvas entry: six source-proved roles required')
    for name in ('width', 'height', 'items', 'source_elements'):
        if type(proof.get(name)) is not int or not 0 < proof[name] <= 16777216:
            raise ValueError('canvas entry: invalid proven extent')
    if (proof['width']*proof['height'] != proof['source_elements'] or
            type(proof.get('tile_items')) is not int or proof['tile_items'] != 256 or proof.get('ownership') != 'copy'):
        raise ValueError('canvas entry: inconsistent proven geometry')
    if not isinstance(proof.get('entry'), str) or not proof['entry'].isidentifier():
        raise ValueError('canvas entry: invalid entry identity')


def checked_module(sources, descriptor):
    validate_proof(descriptor)
    if descriptor['entry'] in {'make_plan', 'execute_plan', '_regular_tile', '_old_tile', '_adora_create_canvas_entry', 'CANVAS_DESCRIPTOR'}:
        raise ValueError('canvas entry name conflicts with checked public API')
    checked, private = {}, {}
    for role in ENTRIES:
        checked[role], private[role] = common.checked_module(sources[role], local_contract(role))
    support = Path(__file__).with_name('prepared_canvas.py').read_text()
    generated = f'''# Compiler-proved private two-loop canvas elimination; no host canvas.
def _adora_create_canvas_entry():
    public = globals()
    get_attribute, binding_error, copy_dict = getattr, ValueError, dict
    descriptor = {descriptor!r}
    sources = {checked!r}
    expected = { {r: hashlib.sha256(s.encode()).hexdigest() for r,s in checked.items()}!r}
    import hashlib
    for role, text in sources.items():
        if hashlib.sha256(text.encode()).hexdigest() != expected[role]:
            raise ValueError('canvas entry: mixed or stale generated kernel')
        ns = {{'__name__': __name__ + '.' + role, '__file__': __file__}}
        exec(compile(text, __file__, 'exec'), ns)
        function = ns[{ENTRIES!r}[role]]
        function._adora_canvas_identity = descriptor['module_identity']
        function._adora_canvas_role = role
        public['_regular_tile' if role == 'regular' else '_old_tile'] = function
    support = {{'__name__': __name__ + '._support', '__file__': __file__}}
    exec(compile({support!r}, __file__, 'exec'), support)
    async def dispatch(role, runtime, *buffers):
        fn = public['_regular_tile' if role == 'regular' else '_old_tile']
        if (get_attribute(fn, '_adora_canvas_identity', None) != descriptor['module_identity'] or
                get_attribute(fn, '_adora_canvas_role', None) != role):
            raise binding_error('canvas entry: mixed or stale kernel binding')
        return await fn(runtime, *buffers)
    async def regular(runtime, *buffers): return await dispatch('regular', runtime, *buffers)
    async def old(runtime, *buffers): return await dispatch('old-only', runtime, *buffers)
    make_plan, execute_plan, once = support['bind_canvas_entry'](regular, old, descriptor)
    public['make_plan'], public['execute_plan'], public[{descriptor['entry']!r}] = make_plan, execute_plan, once
    public['CANVAS_DESCRIPTOR'] = copy_dict(descriptor)
_adora_create_canvas_entry()
del _adora_create_canvas_entry
'''
    compile(generated, '<canvas entry>', 'exec')
    return generated, private


def run_profile(args, require_tool, require_local, strip_module_attrs):
    if args.input.suffix.lower() != '.c' or not all((args.contract, args.adg_path, args.op_file, args.output)):
        raise ValueError('guarded-canvas-mask requires ordinary C, contract, ADG, operations and output')
    if args.enable_unroll:
        raise ValueError('canvas entry does not enable generic unrolling')
    source, output = args.input.resolve(), args.output.resolve()
    if output.suffix != '.py' or output.exists():
        raise ValueError('checked canvas output must be a new .py file')
    contract = common.validate_contract(json.loads(args.contract.read_text()), allow_canvas=True)
    adg, ops = json.loads(args.adg_path.read_text()), json.loads(args.op_file.read_text())
    bank = common.validate_hardware(adg, ops, local_contract('regular'))
    root = args.work_dir.resolve()/'adora-c-entry'
    root.mkdir(parents=True, exist_ok=False)
    frontend = root/'frontend'; frontend.mkdir()
    for src, dst in ((source, frontend/'input.c'), (args.contract, root/'contract.original.json'),
                     (args.adg_path, root/'adg.json'), (args.op_file, root/'operations.json')):
        shutil.copy2(src, dst)
    normalized = root/'contract.json'; normalized.write_text(json.dumps(contract, indent=2)+'\n')
    tools = {n: require_tool(n) for n in ('clang', 'cgeist')}
    tools.update({n: require_local(n) for n in ('cgra-opt', 'cgra-mapper')})
    namefile = os.environ.get('GeneralOpNameFile')
    if not namefile or not Path(namefile).is_file():
        raise ValueError('set GeneralOpNameFile to the compiler operation name table')
    shutil.copy2(namefile, root/'GeneralOpName.txt')
    env = dict(os.environ, GeneralOpNameFile=str(root/'GeneralOpName.txt'))
    provenance = dict(profile='guarded-canvas-mask', status='running', source=dict(path=str(source), sha256=common._sha256(source)),
        tools={n: dict(path=str(Path(p).resolve()), sha256=common._sha256(Path(p))) for n,p in tools.items()}, commands=[],
        production_python={n: dict(sha256=common._sha256(Path(__file__).with_name(n))) for n in
                           ('adoracc.py', 'c_entry.py', 'canvas_entry.py', 'prepared_canvas.py')}, kernels={})
    manifest = root/'provenance.json'
    def save(): manifest.write_text(json.dumps(provenance, indent=2)+'\n')
    def run(command, stage, cwd=root, stdout=None):
        log = root/(stage+'.log')
        record = dict(stage=stage, argv=list(map(str,command)), cwd=str(cwd), log=str(log));provenance['commands'].append(record);save()
        print('+', ' '.join(record['argv']), flush=True)
        with log.open('w') as errors:
            output_stream = stdout.open('w') if stdout else errors
            try:
                process = subprocess.Popen(record['argv'],cwd=cwd,env=env,stdout=output_stream,stderr=errors,start_new_session=True)
                try: process.wait(timeout=args.mapper_wall_timeout if 'mapper' in stage else 300)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait();record['timeout']=True
                record['returncode']=process.returncode;save()
                if process.returncode: raise ValueError(stage+' failed; see '+str(log))
            finally:
                if stdout: output_stream.close()
    pending_caches=[]
    try:
        widths=frontend/'target_widths.h'
        widths.write_text('_Static_assert(__CHAR_BIT__==8 && sizeof(short)==2 && sizeof(int)==4 && sizeof(long)==8 && sizeof(void*)==8,"checked target widths");\n')
        pre=frontend/'preprocessed.c'; tree=frontend/'source.ast.json'
        run([tools['clang'],'-x','c','-std=c11','-target',common.TARGET,'-include',widths,
             *[f'-D{k}={v}' for k,v in sorted(contract['defines'].items())],'-E',source], 'preprocess',stdout=pre)
        run([tools['clang'],'-x','c','-std=c11','-target',common.TARGET,'-Xclang','-ast-dump=json','-fsyntax-only',pre], 'source-ast',stdout=tree)
        source_ast=json.loads(tree.read_text());common.validate_source_ast(source_ast,contract,allow_canvas=True)
        text,edits=common.normalize_unsigned_promotions(source_ast,pre.read_text())
        norm=frontend/'normalized.c';norm.write_text(text)
        (frontend/'unsigned-promotions.json').write_text(json.dumps(edits,indent=2)+'\n')
        raw=frontend/'raw.mlir'
        run([tools['cgeist'],norm,'-O2','-S','-std=c11',f'--target={common.TARGET}',f'--function={contract["entry"]}','--show-ast','-o',raw],'cgeist')
        raw_text=raw.read_text()
        if not re.search(r'llvm\.data_layout = "([^"]+)"',raw_text) or f'llvm.target_triple = "{common.TARGET}"' not in raw_text:
            raise ValueError('canvas cgeist target/layout metadata missing')
        bridge=frontend/'bridge-input.mlir';bridge.write_text(strip_module_attrs(raw_text))
        sources={};common_proof=None
        for role in ENTRIES:
            directory=root/role;directory.mkdir();maps=directory/'map';maps.mkdir();graphs=directory/'dfg';graphs.mkdir()
            local=local_contract(role);(directory/'local-contract.json').write_text(json.dumps(local,indent=2)+'\n')
            proof_path=directory/'host-proof.json';ir=directory/'bridged.mlir'
            run([tools['cgra-opt'],bridge,f'--adora-lower-c-entry=contract={normalized} bank-bytes={bank} large-objects=true canvas-mask=true canvas-kernel={role} descriptor={proof_path}',
                 '--verify-each','-o',ir],role+'-proof')
            proof=json.loads(proof_path.read_text());validate_proof(proof);common.validate_large_source_address(source_ast,proof)
            semantic={k:v for k,v in proof.items() if k not in ('selected_kernel',)}
            if common_proof is not None and semantic!=common_proof:
                raise ValueError('canvas kernels have inconsistent source proofs')
            common_proof=semantic
            final=directory/'final.mlir'
            run([tools['cgra-opt'],ir,'--adora-kernel-dfg-gen','--verify-each','-o',final],role+'-dfg',cwd=graphs)
            dots=list(graphs.glob('*_CDFG.dot'))
            if len(dots)!=1 or 'undefined' in dots[0].read_text():raise ValueError('canvas DFG missing or undefined')
            graph=dots[0].read_text();verify_graph(graph)
            key=hashlib.sha256(b'adora-canvas-kernel-v1'+role.encode()+ir.read_bytes()+
                 json.dumps(proof,sort_keys=True).encode()+json.dumps(provenance['tools'],sort_keys=True).encode()+
                 json.dumps(provenance['production_python'],sort_keys=True).encode()+normalized.read_bytes()+
                 (root/'adg.json').read_bytes()+(root/'operations.json').read_bytes()+(root/'GeneralOpName.txt').read_bytes()+
                 repr((args.seed,args.max_iters,args.timeout)).encode()).hexdigest()
            cache=args.kernel_cache.resolve()/('canvas-v1-'+key) if args.kernel_cache else None
            hit=False
            if cache is not None and (cache==root or root in cache.parents):raise ValueError('canvas cache must be outside compilation artifacts')
            if cache is not None and cache.exists():
                if cache.is_symlink() or (cache/'hashes.json').is_symlink():raise ValueError('canvas cache symlink refused')
                cache_manifest=json.loads((cache/'hashes.json').read_text());verify_cache(cache/'map',cache_manifest,key)
                shutil.copytree(cache/'map',maps,dirs_exist_ok=True);hit=True
            unchecked=maps/'unchecked.py'
            if not hit:
                run([tools['cgra-mapper'],final,f'--adg={root / "adg.json"}',f'--op-file={root / "operations.json"}',
                     '--output-type=pytest',f'--output={unchecked}',f'--seed={args.seed}',f'--max-iters={args.max_iters}',f'--timeout={args.timeout}'],role+'-mapper',cwd=maps)
            mapdirs=list(maps.glob('*_map_result'))
            if len(mapdirs)!=1 or not unchecked.is_file() or not (mapdirs[0]/'config.bit').stat().st_size:
                raise ValueError('canvas mapper output incomplete')
            before=list(mapdirs[0].glob('before_map_*_CDFG.dot'))
            if len(before)!=1 or before[0].read_text()!=graph:raise ValueError('canvas mapper consumed a different graph')
            check=verify_graph(graph,(mapdirs[0]/'mapped_dfg.dot').read_text())
            (directory/'operand-verification.json').write_text(json.dumps(check,indent=2)+'\n')
            sources[role]=unchecked.read_text()
            provenance['kernels'][role]=dict(entry=ENTRIES[role],map_directory=str(mapdirs[0].relative_to(root)),cache_key=key,cache_hit=hit)
            if cache is not None and not hit:pending_caches.append((cache,maps,key))
        proof=common_proof;proof['support_version']=1
        proof['module_identity']=hashlib.sha256(frontend.joinpath('input.c').read_bytes()+normalized.read_bytes()+
            json.dumps(proof,sort_keys=True).encode()+json.dumps(sources,sort_keys=True).encode()+
            json.dumps(provenance['tools'],sort_keys=True).encode()+json.dumps(provenance['production_python'],sort_keys=True).encode()).hexdigest()
        generated,private=checked_module(sources,proof)
        for role,text in private.items():(root/role/'map/generated.py').write_text(text)
        (root/'host-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
        for cache,maps,key in pending_caches:
            cache.parent.mkdir(parents=True,exist_ok=True)
            pending=Path(tempfile.mkdtemp(prefix=cache.name+'.pending-',dir=cache.parent))
            shutil.copytree(maps,pending/'map')
            files={str(p.relative_to(pending/'map')):common._sha256(p) for p in (pending/'map').rglob('*') if p.is_file()}
            (pending/'hashes.json').write_text(json.dumps(dict(key=key,files=files),sort_keys=True))
            pending.rename(cache)
        if common._sha256(source)!=provenance['source']['sha256']:raise ValueError('canvas source changed during compilation')
        output.parent.mkdir(parents=True,exist_ok=True)
        provenance.update(status='complete',verified_host_descriptor=proof,
                          checked_entry=dict(path=str(output),sha256=hashlib.sha256(generated.encode()).hexdigest(),name=contract['entry']))
    except Exception as exc:
        provenance.update(status='failed',error=str(exc));raise
    finally:
        try:
            provenance['artifacts']={str(p.relative_to(root)):common._sha256(p) for p in sorted(root.rglob('*')) if p.is_file() and p!=manifest}
            save()
        except Exception as exc:
            provenance.update(status='failed', error=str(exc))
            try:
                save()
            except OSError:
                pass
            raise
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=output.parent,prefix='.adora-canvas-',delete=False) as stream:
            temporary=Path(stream.name);stream.write(generated)
        os.link(temporary,output)
    except Exception as exc:
        provenance.update(status='failed',error=str(exc));save();raise
    finally:
        if temporary is not None:
            try:temporary.unlink()
            except OSError:pass
    print(f'[adoracc] Checked canvas entry: {output}\n[adoracc] Provenance: {manifest}')
    return 0
