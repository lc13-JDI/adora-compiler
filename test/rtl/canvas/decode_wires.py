"""Offline decoder for little-endian 28-byte canvas wire records, independent of cocotb."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from canvas_prepare import daily_frames, oracle

RECORD = struct.Struct('<IBBHI4I')
DATA = struct.Struct('<4I')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def frame_source(path, contract):
    if not path:
        for arrays in daily_frames(contract):
            yield arrays, oracle(arrays, contract['defines'])[0]
        return
    with np.load(path, allow_pickle=False) as file:
        for i in range(int(file['frames'])):
            yield tuple(file[f'{name}_{i}'] for name in ('src', 'x', 'y', 'enabled', 'old')), file[
                f'gold_{i}'
            ]


def partitions(events):
    current = None
    resets = []
    for row in events:
        kind = row[1]
        if kind == 10:
            resets.append(row[2])
        if kind == 16:
            require(current is None, 'nested call')
            current = []
        if current is not None:
            current.append(row)
        elif kind < 10 or kind == 18:
            raise ValueError('unattributed bus or wait activity')
        if kind == 17:
            require(current is not None, 'end without begin')
            yield current
            current = None
    require(current is None and resets == [0, 1, 0], 'incomplete call or reset sequence')


def payload(row):
    return DATA.pack(*row[5:9])


def decode_call(rows, frame, block, arrays, guard, gold, meta):
    begin, end = rows[0], rows[-1]
    require(begin[3:5] == end[3:5] == (frame, block), 'wrong frame/block order')
    kernel = 'regular' if guard.any() else 'old-only'
    binding = meta['kernels'][kernel]
    roles = binding['roles']
    by_kind = [[] for _ in range(19)]
    for row in rows:
        by_kind[row[1]].append(row)
    offset = block * 256
    old = np.zeros(256, dtype='<u2')
    old[: len(guard)] = arrays[-1][offset : offset + len(guard)]
    mask = np.zeros(256, dtype='<u2')
    mask[: len(guard)] = gold[offset : offset + len(guard)]
    expected = {'old': old.tobytes(), 'config': bytes.fromhex(binding['config_hex'])}
    if kernel == 'regular':
        positions = np.flatnonzero(guard)
        packed = np.zeros(256, dtype='<u2')
        width = meta['contract']['defines']['W']
        addr = (
            arrays[2][offset + positions].astype(np.int64) * width + arrays[1][offset + positions]
        )
        packed[positions] = arrays[0][addr]
        enabled = np.zeros(256, dtype='<i2')
        enabled[positions] = 1
        expected.update(source=packed.tobytes(), enabled=enabled.tobytes())
        if 'index' in roles:
            index = np.full(256, -1, dtype='<i2')
            index[positions] = positions
            expected['index'] = index.tobytes()
    addresses = {r['address']: name for name, r in roles.items()}
    addresses[meta['cfg_base']] = 'config'
    writes = by_kind[0]
    responses = by_kind[2]
    require(len(writes) == len(responses) == len(expected), 'write address/response count')
    wpos = 0
    totals = Counter()
    spans = {}
    for aw, b in zip(writes, responses):
        name = addresses.get(aw[4])
        require(name in expected, 'unexpected or repeated transfer')
        n = aw[3] + 1
        beats = by_kind[1][wpos : wpos + n]
        wpos += n
        require(
            aw[2] == 1 and aw[5] == 4 and len(beats) == n and b[4] == aw[6] << 8,
            'AXI burst or response',
        )
        require(aw[0] <= beats[0][0] <= beats[-1][0] < b[0], 'write timing')
        require([w[2] for w in beats] == [0] * (n - 1) + [1], 'WLAST')
        received = bytearray()
        for w in beats:
            received.extend(v for j, v in enumerate(payload(w)) if w[3] & (1 << j))
        require(received == expected.pop(name), 'accepted ' + str(name) + ' bytes mismatch')
        totals[name + '_bytes'] += len(received)
        totals[name + '_occupied_bytes'] += 16 * n
        spans[name] = (aw[0], b[0])
    require(not expected and wpos == len(by_kind[1]), 'unconsumed write data')
    (ar,) = by_kind[3]
    r = by_kind[4]
    require(
        ar[4] == roles['mask']['address'] and ar[3:4] == (31,) and ar[2] == 1 and ar[5] == 4,
        'D2H address/burst',
    )
    require(
        len(r) == 32 and [x[2] for x in r] == [0] * 31 + [1] and all(x[4] == ar[6] << 8 for x in r),
        'D2H responses',
    )
    data = b''.join(payload(b) for b in r)
    require(data == mask.tobytes(), 'raw D2H differs from oracle or padding')
    require(
        len(by_kind[5]) == len(by_kind[6]) == len(by_kind[7])
        and len(by_kind[8]) == len(by_kind[9]),
        'control pairing',
    )
    controls = [(aw[4], w[3], w[5]) for aw, w in zip(by_kind[5], by_kind[6])]
    require(
        [a for a, s, v in controls] == [0, 4, 8, 12, 16, 20, 24, 28] and controls[-1][2] == 1,
        'control START/config sequence',
    )
    require(
        all(
            aw[0] <= w[0] <= b[0] and not b[4]
            for aw, w, b in zip(by_kind[5], by_kind[6], by_kind[7])
        ),
        'control write response',
    )
    require(
        all(a[4] == 32 and a[0] <= b[0] and not b[4] for a, b in zip(by_kind[8], by_kind[9])),
        'control read channel',
    )
    totals.update(
        control_write_bytes=sum(s.bit_count() for a, s, v in controls),
        control_read_bytes=4 * len(by_kind[9]),
        d2h_bytes=len(data),
        calls=1,
    )
    (config_ready,) = [x[0] for x in by_kind[11] if x[2]]
    starts = [x for x in by_kind[12] if x[3]]
    require(len(starts) == 1, 'physical start count')
    start = starts[0][0]
    require(config_ready <= start and starts[0][3] == controls[6][2], 'configured tile starts')
    (wait,) = by_kind[18]
    require(
        wait[4] & wait[3] == wait[3] and start < wait[5] <= wait[0] < ar[0],
        'fresh runtime completion',
    )
    require(
        any(x[0] == wait[5] and x[5] & wait[3] == wait[3] for x in by_kind[9])
        or any(start < x[0] <= wait[5] and x[5] & wait[3] == wait[3] for x in by_kind[9]),
        'no accepted fresh AXIL done',
    )
    require(
        any(x[0] >= start and x[4] & wait[3] != wait[3] for x in by_kind[12]),
        'no physical busy state',
    )
    done = next(x[0] for x in by_kind[12] if x[0] > start and x[4] & wait[3] == wait[3])
    physical = [x for x in by_kind[13] if x[2]]
    output = roles['mask']
    require(
        len(physical) == 256
        and all(
            x[2:5] == (3, output['bank'], output['word_offset'] + i) and x[5] == mask[i]
            for i, x in enumerate(physical)
        ),
        'SRAM write stream mismatch',
    )
    require(
        physical[-1][0] <= done <= wait[0] and physical[-1][0] < ar[0],
        'last write completion order',
    )
    if begin[2]:
        for name, role in roles.items():
            if name == 'mask':
                continue
            reads = [
                x[4] - role['word_offset'] for x in by_kind[13] if x[3] == role['bank'] and not x[2]
            ]
            expect = list(range(256))
            if name == 'source' and 'index' in roles:
                expect = [i if i < len(guard) and guard[i] else 0 for i in range(256)]
            require(reads == expect, 'physical ' + name + ' read addresses mismatch')
    h2d_start = min(t[0] for n, t in spans.items() if n != 'config')
    h2d_end = max(t[1] for n, t in spans.items() if n != 'config')
    endpoints = [
        begin[0],
        spans['config'][0],
        spans['config'][1],
        h2d_start,
        h2d_end,
        start,
        done,
        ar[0],
        r[-1][0],
        end[0],
    ]
    require(endpoints == sorted(endpoints), 'phase chronology')
    for name, a, b in zip(
        (
            'entry',
            'config_dma',
            'config_control',
            'app_h2d',
            'launch',
            'execute',
            'finish_wait',
            'd2h',
            'copy_return',
        ),
        endpoints,
        endpoints[1:],
    ):
        totals[name + '_cycles'] += b - a
    totals.update(
        interface_cycles=end[0] - begin[0],
        cfg_active_overlap_h2d_cycles=max(
            0, min(config_ready, h2d_end) - max(h2d_start, by_kind[6][3][0])
        ),
    )
    return kernel, totals, data[: len(guard) * 2]


def decode(wire_path, meta, frames):
    raw = Path(wire_path).read_bytes()
    return decode_buffer(raw, meta, frames)


def decode_buffer(raw, meta, frames):
    chunks = partitions(RECORD.iter_unpack(raw))
    results = []
    for i, (arrays, gold) in enumerate(frames):
        _, guard = oracle(arrays, meta['contract']['defines'])
        totals = {}
        h = hashlib.sha256()
        for b, offset in enumerate(range(0, len(guard), 256)):
            rows = next(chunks, None)
            require(rows is not None, 'missing call')
            kernel, counts, raw_output = decode_call(
                rows, i, b, arrays, guard[offset : offset + 256], gold, meta
            )
            totals.setdefault(kernel, Counter()).update(counts)
            h.update(raw_output)
        results.append(
            dict(frame=i, kernels=totals, public_words=len(gold), raw_public_sha256=h.hexdigest())
        )
    require(next(chunks, None) is None, 'extra call')
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--frames', type=Path)
    p.add_argument('--tamper-test', action='store_true')
    args = p.parse_args()
    meta = json.loads((args.run / 'bindings.json').read_text())
    frames = list(frame_source(args.frames, meta['contract']))
    raw = (args.run / 'wire.bin').read_bytes()
    summary = decode_buffer(raw, meta, frames)
    rejected = {}
    if args.tamper_test:
        first = {}
        for i, row in enumerate(RECORD.iter_unpack(raw)):
            first.setdefault(row[1], i * RECORD.size)
            if 17 in first:
                break
        for typ in ('drop_call', 'change_byte', 'change_frame', 'change_block'):
            changed = bytearray(raw)
            if typ == 'drop_call':
                del changed[first[16] : first[17] + RECORD.size]
            elif typ == 'change_byte':
                changed[first[4] + 12] ^= 1
            else:
                changed[first[16] + (6 if typ == 'change_frame' else 8)] ^= 1
            try:
                decode_buffer(changed, meta, frames)
            except (ValueError, StopIteration) as err:
                rejected[typ] = str(err)
            else:
                raise ValueError('undetected ' + typ)
    print(
        json.dumps(
            dict(
                frames=summary, tamper_rejections=rejected, sha256=hashlib.sha256(raw).hexdigest()
            ),
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
