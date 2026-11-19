#include <stdint.h>
#include <stddef.h>

_Static_assert(W > 0 && H > 0 && T > 0, "positive fixed bounds");
_Static_assert(0 < LO && LO <= HI && HI <= UINT16_MAX, "uint16 depth interval");

void depth_canvas_mask(const uint16_t *src, const int16_t *x, const int16_t *y,
                       const int16_t *enabled, const uint16_t *old, uint16_t *mask) {
    uint16_t canvas[T];
    for (int i = 0; i < T; ++i) {
        int xx = x[i], yy = y[i];
        canvas[i] = old[i];
        if (enabled[i] != 0 && xx >= 0 && xx < W && yy >= 0 && yy < H) {
            uint16_t depth = src[(size_t)yy * W + (size_t)xx];
            canvas[i] = LO <= depth && depth <= HI ? depth : 0;
        }
    }
    for (int i = 0; i < T; ++i)
        mask[i] = canvas[i] != 0 ? 255 : 0;
}
