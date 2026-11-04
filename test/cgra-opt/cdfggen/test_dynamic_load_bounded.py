import os, pathlib, re, subprocess, sys, tempfile
opt, source = sys.argv[1:]
text = pathlib.Path(source).read_text()
cases = [("valid", text, True), ("missing", text.replace('adora.value_range = [0 : i64, 3 : i64], ', ''), False), ("outside", text.replace('3 : i64]', '4 : i64]'), False), ("negative", text.replace('[0 : i64, 3', '[-1 : i64, 3'), False), ("badtype", text.replace('[0 : i64, 3 : i64]', '[0 : i32, 3 : i32]'), False), ("multitrip", text.replace('= 0 to 1', '= 0 to 2'), False)]
host_write = '    %bad = arith.constant 4 : i16\n    affine.store %bad, %arg1[%arg3] : memref<2xi16>\n'
output_copy = next(line for line in text.splitlines() if 'BlockStore %1,' in line)
cases += [
 ("memory-space", text.replace('memref<8xi16>', 'memref<8xi16, 1>'), False),

 ("kernel-call", text.replace('module {', 'module {\n  func.func private @mutate(memref<2xi16>)', 1).replace('    ADORA.kernel {', '    ADORA.kernel {\n      func.call @mutate(%arg1) : (memref<2xi16>) -> ()'), False),
 ("kernel-controlflow", text.replace('    %c1 =', '    %true = arith.constant true\n    %c1 =').replace('    ADORA.kernel {', '    ADORA.kernel {\n      scf.if %true {\n      }'), False),

 ("host-affine-write", text.replace('    %c1 =', host_write + '    %c1 ='), False),
 ("host-memref-write", text.replace('    %c1 =', host_write.replace('affine.store', 'memref.store') + '    %c1 ='), False),
 ("output-before-kernel", text.replace(output_copy + '\n', '').replace('    ADORA.kernel {', output_copy + '\n    ADORA.kernel {'), False),
 ("host-loop", text.replace('    ADORA.kernel {', '    affine.for %host_iv = 0 to 1 {\n    ADORA.kernel {').replace('    return', '    }\n    return'), False),

 ("output-store-offset", text.replace('affine.store %15, %1[0]', 'affine.store %15, %1[1]'), False),
 ("output-extra-write", text.replace('affine.store %15, %1[0]', 'affine.store %15, %1[0] : memref<1xi16>\n        affine.store %15, %1[0]'), False),
 ("output-emitskip", text.replace('Id = "out1", KernelName', 'EmitSkip, Id = "out1", KernelName'), False),
 ("output-oob", text.replace('%1, %arg2 [1]', '%1, %arg2 [2]'), False),
 ("output-dynamic", text.replace('%1, %arg2 [1]', '%1, %arg2 [%arg3]'), False),
 ("output-missing", '\n'.join(line for line in text.splitlines() if 'BlockStore %1,' not in line), False),
 ("output-overlap", text.replace('%1, %arg2 [1]', '%1, %arg2 [0]'), False),
 ("output-size", text.replace('memref<1xi16>', 'memref<2xi16>'), False),
 ("output-id", text.replace('{Id = "out1", KernelName = "ebag_plain"}\n    return', '{Id = "wrong", KernelName = "ebag_plain"}\n    return'), False),

 ("boundary-source", text.replace('memref.load %result[', 'memref.load %arg0['), False),
 ("nonidentity", text.replace('memref<8xi16>', 'memref<8xi16, strided<[1], offset: 1>>'), False),
 ("wrong-kernel", text.replace('Id = "w", KernelName = "ebag_plain"', 'Id = "w", KernelName = "other"'), False),
 ("local-object", text.replace('%result = ADORA.BlockLoad %arg0 [0] : memref<8xi16> -> memref<8xi16>', '%result = ADORA.LocalMemAlloc memref<8xi16>'), False),
 ("overflow", text.replace('arith.constant 2 : index', 'arith.constant 65536 : index'), False),
 ("duplicate-id", text.replace('Id = "w"', 'Id = "ids"'), False),
 ("mutable-object", text.replace('        %3 = arith.index_cast', '        affine.store %2, %result[0] : memref<8xi16>\n        %3 = arith.index_cast'), False),
 ("host-offset", text.replace('BlockLoad %arg0 [0]', 'BlockLoad %arg0 [1]'), False),
 ("bad-domain-order", text.replace('[0 : i64, 3 : i64]', '[3 : i64, 0 : i64]'), False),
 ("bad-domain-width", text.replace('[0 : i64, 3 : i64]', '[0 : i64, 32768 : i64]'), False),
]
for name, ir, success in cases:
 with tempfile.TemporaryDirectory(prefix='adora-load-') as tmp:
  path = pathlib.Path(tmp)/'input.mlir'; path.write_text(ir)
  result = subprocess.run([opt, str(path), '--adora-kernel-dfg-gen'], cwd=tmp, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  assert (result.returncode == 0) == success, name + ': ' + result.stdout
  if success:
   dot = (pathlib.Path(tmp)/'ebag_plain_CDFG.dot').read_text()
   loads = [line for line in dot.splitlines() if 'opcode = "load"' in line]
   assert len(loads) == 4, loads
   assert all('ref_name="ebag_plain:w"' in line and 'size="16"' in line and 'pattern="0,1"' in line and 'offset="0,0"' in line for line in loads), loads
   edges = re.findall(r'^(\w+) -> (\w+)\[.*operand = (-?\d+)', dot, re.M)
   nodes = dict(re.findall(r'^(\w+)\[(.*)\];$', dot, re.M))
   for line in loads:
    name = line.split('[')[0]
    incoming = [(a, int(port)) for a, b, port in edges if b == name]
    assert len(incoming) == 1 and incoming[0][1] == 0, incoming
    scale = incoming[0][0]
    assert 'opcode = "MUL"' in nodes[scale], nodes[scale]
    constants = [a for a, b, port in edges if b == scale and port == '1']
    assert len(constants) == 1 and 'value="0x00000002"' in nodes[constants[0]], constants
  else:
   assert 'bounded dynamic LOAD' in result.stdout, result.stdout
print('bounded dynamic LOAD: all cases passed')
