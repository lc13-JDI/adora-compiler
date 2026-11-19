/* R010 ordinary global C input, not a hand-written tile kernel.
 * W, H, T and VALUE_NONZERO are explicit positive compile-time contract defines.
 * Keep the global loop/address arithmetic wide on the HOST. No device int32 claim.
 */
#include <stdint.h>
#include <stddef.h>
#ifndef W
#define W 640
#endif
#ifndef H
#define H 480
#endif
#ifndef T
#define T 307200
#endif
#ifndef VALUE_NONZERO
#define VALUE_NONZERO 0
#endif
_Static_assert(sizeof(uint16_t)==2 && sizeof(int16_t)==2 && sizeof(int32_t)==4, "widths");
_Static_assert(W>0 && W<=32767 && H>0 && H<=32767 && T>0 && T<=2147483647, "global bounds");
void large_overlay(const uint16_t *src, const int16_t *x, const int16_t *y,
                   const int16_t *enabled, uint16_t *dst) {
  for (int32_t i = 0; i < T; ++i) {
    int32_t xx = x[i], yy = y[i];
    if (enabled[i] != 0 && xx >= 0 && yy >= 0 && xx < W && yy < H) {
      uint16_t value = src[(size_t)yy * (size_t)W + (size_t)xx];
#if VALUE_NONZERO
      if (value != 0) dst[i] = value;
#else
      dst[i] = value;
#endif
    }
  }
}
