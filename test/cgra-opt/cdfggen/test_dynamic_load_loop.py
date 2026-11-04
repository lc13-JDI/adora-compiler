import pathlib, subprocess, sys, tempfile, re
opt, source = sys.argv[1:]
text = pathlib.Path(source).read_text()
for length in (4, 1):
 with tempfile.TemporaryDirectory(prefix="adora-load-loop-") as tmp:
  path = pathlib.Path(tmp)/"input.mlir"
  path.write_text(text.replace("= 0 to 4", f"= 0 to {length}"))
  result = subprocess.run([opt, str(path), "--adora-kernel-dfg-gen"], cwd=tmp, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  assert result.returncode == 0, result.stdout
  dot = (pathlib.Path(tmp)/"sum_l4_n4_CDFG.dot").read_text()
  for opcode, count in (("Input",1),("load",2),("ACC",2),("Output",2)):
   lines = [line for line in dot.splitlines() if f'opcode = "{opcode}"' in line]
   assert len(lines) == count, (opcode, lines, dot)
   if opcode == "load": assert all(f'pattern="0,{length}"' in line for line in lines), lines
   if opcode == "Output": assert all('pattern="0,1"' in line for line in lines), lines
   if opcode == "ACC": assert all(f'acc_params="0,{length},1,1"' in line for line in lines), lines
   if opcode == "Input": assert f'pattern="2,{length}"' in lines[0], lines
  nodes = dict(re.findall(r'^(\w+)\[(.*)\];$', dot, re.M))
  edges = re.findall(r'^(\w+) -> (\w+)\[.*operand = (-?\d+)', dot, re.M)
  for node, attrs in nodes.items():
   incoming = [(src, int(port)) for src, dst, port in edges if dst == node]
   if 'opcode = "load"' in attrs:
    assert len(incoming) == 1 and incoming[0][1] == 0, incoming
    scale = incoming[0][0]
    assert 'opcode = "MUL"' in nodes[scale]
    constants = [src for src, dst, port in edges if dst == scale and port == '1']
    assert len(constants) == 1 and 'value="0x00000002"' in nodes[constants[0]]
   if 'opcode = "Output"' in attrs:
    assert len(incoming) == 1 and incoming[0][1] == 0 and 'opcode = "ACC"' in nodes[incoming[0][0]], incoming
cases = {
 "constant-index": text.replace('affine.load %ids[%j]', 'affine.load %ids[0]'),
 "stray-iv": text.replace('    %id =', '    %stray = arith.addi %j, %c1 : index\n    %id ='),
 "stray-arithmetic": text.replace('    %id =', '    %stray = arith.addi %c1, %c2 : index\n    %id ='),
 "unknown-range": text.replace(', adora.value_range = [0 : i64, 3 : i64]', ''),
 "oob": text.replace('3 : i64]', '4 : i64]'),
 "empty": text.replace('= 0 to 4', '= 0 to 0'),
 "negative": text.replace('= 0 to 4', '= 0 to -1'),
 "step": text.replace('to 4 iter_args', 'to 4 step 2 iter_args'),
 "initial": text.replace('arith.constant 0 : i16', 'arith.constant 1 : i16'),
 "carry": text.replace('%n1 = arith.addi %s1, %y', '%n1 = arith.addi %s0, %y'),
 "mutable": text.replace('    %id =', '    affine.store %zero, %w[0] : memref<8xi16>\n    %id ='),
 "intermediate": text.replace('    affine.yield', '    affine.store %n0, %out0[0] : memref<1xi16>\n    affine.yield'),
 "memref-index": text.replace('%id = affine.load', '%id = memref.load'),
 "dynamic-bound": text.replace('= 0 to 4 iter_args', '= 0 to %arg3 iter_args'),
 "nested": text.replace('    %id =', '    affine.for %k = 0 to 1 {}\n    %id ='),
}
for name, ir in cases.items():
 with tempfile.TemporaryDirectory(prefix="adora-load-reject-") as tmp:
  path = pathlib.Path(tmp)/"input.mlir"; path.write_text(ir)
  result = subprocess.run([opt, str(path), "--adora-kernel-dfg-gen"], cwd=tmp, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  assert result.returncode != 0 and "bounded dynamic LOAD" in result.stdout, (name, result.stdout)
print("bounded dynamic reduction loop: passed L4/L1 and rejection cases")
