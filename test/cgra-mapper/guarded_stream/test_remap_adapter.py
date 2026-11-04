"""The sample adapter preserves words and does not dispatch empty maps."""
import asyncio
import numpy as np
from remap_u16 import remap_u16

async def test():
    src = np.array([0,32767,32768,60000,65535], dtype=np.uint16)
    mx = np.array([-32768,-1,0,1,32767], dtype=np.int16)
    my = mx.copy(); dst = np.zeros(5, dtype=np.uint16)
    calls = []
    async def kernel(runtime, s, x, y, out):
        calls.append(runtime)
        assert s.dtype == out.dtype == np.int16
        assert np.shares_memory(s, src) and np.shares_memory(out, dst)
        assert np.array_equal(x, mx) and np.array_equal(y, my)
        # Adapter-only test: this stub is NOT an execution oracle.
        out[:] = s
    await remap_u16('sentinel', kernel, src, mx, my, dst)
    assert calls == ['sentinel'] and np.array_equal(dst, src)
    await remap_u16('sentinel', kernel, src, mx[:0], my[:0], dst[:0])
    assert len(calls) == 1
    for args in [(src[:0],mx,my,dst),(src,mx.astype(np.uint16),my,dst),
                 (src,mx,my,src),(src,mx,my,dst.astype(np.int16))]:
        try: await remap_u16('sentinel',kernel,*args)
        except ValueError: pass
        else: raise AssertionError('bad adapter input accepted')
    assert len(calls) == 1
asyncio.run(test())
print('u16 adapter: word views, signed coordinates, empty no-dispatch and invalid inputs PASS')
