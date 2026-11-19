"""Execute the compiler-produced graph at the 16-bit datapath boundary."""
import functools
import re


def check_regular_graph(text, variant, lo, hi):
    stream = variant in ('C', 'D')
    nodes = {name: dict(re.findall(r'(\w+)\s*=\s*"([^"]*)"', attrs))
             for name, attrs in re.findall(r'(?m)^(\w+)\[([^\n]*)\];', text)}
    incoming = {}
    for source, dest, port in re.findall(r'(?m)^(\w+) -> (\w+)\[.*?operand = (\d+)', text):
        incoming.setdefault(dest, {})[int(port)] = source
    roles = [n for n in nodes.values() if 'ref_name' in n]
    assert len(roles) == (4 if stream else 5)
    for node in roles:
        address_stream = node['ref_name'].endswith(':arg0') and not stream
        assert node['opcode'].lower() == ('load' if address_stream else
                                          'output' if node['ref_name'].endswith(':arg' + str(len(roles)-1)) else 'input')
        assert node['pattern'] == ('0,256' if address_stream else '2,256')
    assert all(node['opcode'] != 'MUL' for node in nodes.values()) == stream
    output, = [name for name, n in nodes.items() if n['opcode'] == 'Output']

    def evaluate(buffers, slot):
        def signed(value):
            return value if value < 32768 else value - 65536

        @functools.lru_cache(None)
        def evaluate_node(name):
            node = nodes[name]
            operands = [evaluate_node(source) for _, source in sorted(incoming.get(name, {}).items())]
            op = node['opcode']
            if op in ('Input', 'load'):
                arg = int(node['ref_name'].split(':arg')[1])
                if op == 'Input':
                    return buffers[arg][slot] & 65535
                address, = operands
                assert address % 2 == 0 and address < 512
                return buffers[arg][address // 2] & 65535
            if op == 'Output':
                return operands[0]
            if op == 'CONST':
                return int(node['value'], 0) & 65535
            if op == 'SEL':
                no, yes, condition = operands
                assert condition in (0, 1)
                return yes if condition else no
            a, b = operands
            return {'AND': a & b, 'OR': a | b, 'ADD': (a + b) & 65535,
                    'MUL': (a * b) & 65535, 'SLT': int(signed(a) < signed(b)),
                    'SLE': int(signed(a) <= signed(b))}[op]
        return evaluate_node(output)

    samples = sorted({value for value in (0, 1, lo-1, lo, lo+1, hi-1, hi, hi+1, 32768, 65535)
                      if 0 <= value <= 65535})
    enabled, packed, old = [0]*256, [0]*256, [0]*256
    index = list(range(256))
    arrays = [packed, enabled, old] if stream else [packed, index, enabled, old]
    for slot in (0, 255):
        for previous in (0, 32768, 65535):
            old[slot] = previous
            for value in samples:
                packed[slot] = value
                for active in (0, 1):
                    enabled[slot] = active
                    canvas = (value if lo <= value <= hi else 0) if active else previous
                    assert evaluate(arrays, slot) == (255 if canvas else 0)

    # Research-only permutation of the private packed representation. Its
    # explicit index undoes the swap in indexed arms. Streaming now reads
    # the opposite threshold classes at both endpoints.
    enabled[0] = enabled[255] = 1
    packed[0], packed[255] = 0, lo
    index[0], index[255] = 255, 0
    swapped_output = [evaluate(arrays, s) for s in (0, 255)]
    assert swapped_output == ([0, 255] if stream else [255, 0])
    # With LO=0, scalar source depth zero passes the interval and retains a
    # zero canvas. Its consumer disagrees with a direct interval predicate.
    depth = 0
    admitted = 0 <= depth <= hi
    scalar_canvas = depth if admitted else 0
    consumer = 255 if scalar_canvas else 0
    direct = 255 if admitted else 0
    assert consumer == 0 and direct == 255
    return dict(arm=variant, range=[lo, hi], checked_depth_codes=samples,
                slots=[0, 255], logical_depths=[lo, 0], swapped_packed=[0, lo],
                swapped_index=[255, 0], swapped_output=swapped_output,
                zero_bound_counterexample=dict(lo=0, hi=hi, depth=depth,
                                               original=consumer, direct=direct))
