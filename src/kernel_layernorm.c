#include "kernel.h"
#include "model.h"
#include <stdint.h>
#include <stddef.h>

#define ONE_Q15 32768

static uint32_t rsqrt_q15(uint32_t s2_q15, uint8_t epsilon_shift)
{
    s2_q15 += (ONE_Q15 >> epsilon_shift);

    uint32_t y = 1 << 30;

    for (int i = 0; i < 4; ++i) {
        uint64_t y2   = ((uint64_t)y * y) >> 30;           /* y²  Q0.30 */
        uint64_t prod = ((uint64_t)s2_q15 << 15) * y2;     /* s²*y² Q0.45 */
        uint32_t term = (uint32_t)(prod >> 30);            /* Q0.15 */
        uint32_t half = term >> 1;                         /* 0.5*s²*y² */
        uint32_t tmp  = (3 * ONE_Q15 / 2) - half;          /* 1.5 − … */
        y = (uint32_t)(((uint64_t)y * tmp) >> 15);         /* Q0.30 */
    }

    return y >> 15;                                        /* Q1.15 */
}

void layernorm_int8(qint8_t        *x,
                    const qint32_t *gamma,
                    const qint32_t *beta,
                    size_t          len,
                    uint8_t         epsilon_shift)
{

    int64_t sum  = 0;
    int64_t sum2 = 0;
    for (size_t i = 0; i < len; ++i) {
        int32_t v = x[i];
        sum  += v;
        sum2 += v * v;
    }

    int32_t mean_q15 = (int32_t)((sum << 15) / (int64_t)len);
    int64_t ex2_q15   = (sum2 << 15) / (int64_t)len;
    int64_t mean_sq   = ((int64_t)mean_q15 * mean_q15) >> 15;

    int32_t s2_q15    = (int32_t)(ex2_q15 - mean_sq);
    if (s2_q15 < 0) s2_q15 = 0;

    uint32_t inv_sigma_q15 = rsqrt_q15((uint32_t)s2_q15, epsilon_shift);

    for (size_t i = 0; i < len; ++i) {
        int32_t v_q15 = ((int32_t)x[i] << 15) - mean_q15;
        int32_t n_q15 = (int32_t)(((int64_t)v_q15 * inv_sigma_q15) >> 15);
        int32_t scaled = (int32_t)(((int64_t)n_q15 * gamma[i]) >> 15) + beta[i];
        int32_t y = (scaled + (1 << 14)) >> 15;
        x[i] = sat8(y);
    }
}
