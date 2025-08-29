#include "kernel.h"
#include "model.h"
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

#define Q15(x)   ((int32_t)((x) * 32768.0f + ((x) >= 0 ? 0.5f : -0.5f)))

static const int32_t ONE_Q15   = 32768;             /* 1.0 */
static const int32_t LN2_Q15   = Q15(0.693147f);    /* ln 2  ≈ 0.6931 */
static const int32_t C0_Q15    = ONE_Q15;           /* c0 = 1 */
static const int32_t C1_Q15    = LN2_Q15;           /* c1 = ln2 */
static const int32_t C2_Q15    = Q15(0.240226f);    /* c2 = (ln2²)/2 */

static inline qint8_t sat8_pos(uint32_t x)
{
    return (x > 127u) ? 127 : (qint8_t)x;
}

static uint16_t exp2_poly_q15(int16_t diff)
{

    uint16_t d = (uint16_t)(-diff);
    uint8_t k = (uint8_t)(d >> 5);
    uint8_t r = (uint8_t)(d & 31);

    int32_t x_q15 = -((int32_t)r << 10);

    int32_t t = (int32_t)(((int64_t)C2_Q15 * x_q15) >> 15);
    t = (t + C1_Q15);
    t = (int32_t)(((int64_t)t * x_q15) >> 15);
    int32_t p_q15 = t + C0_Q15;

    uint32_t base_q15 = ONE_Q15 >> k;
    uint32_t res_q15  = (uint32_t)(((uint64_t)base_q15 * p_q15) >> 15);
    return (uint16_t)res_q15;
}

void softmax_int8(const qint8_t *x, qint8_t *y, size_t len)
{
    if (len == 0) return;
    int8_t maxv = x[0];
    for (size_t i = 1; i < len; ++i)
        if (x[i] > maxv) maxv = x[i];

    #define STACK_BUF_MAX 256
    uint16_t stack_buf[STACK_BUF_MAX];
    uint16_t *buf = (len <= STACK_BUF_MAX) ? stack_buf : (uint16_t *)malloc(len * sizeof(uint16_t));
    if (!buf) { 
        for (size_t i = 0; i < len; ++i) 
            y[i] = (qint8_t)(127 / (int)len);
        return;
    }

    uint32_t sum_q15 = 0;

    for (size_t i = 0; i < len; ++i) {
        int16_t diff = (int16_t)x[i] - maxv;
        if (diff < -255) diff = -255;
        uint16_t e_q15 = exp2_poly_q15(diff);
        buf[i] = e_q15;
        sum_q15 += e_q15;
    }

    for (size_t i = 0; i < len; ++i) {
        uint32_t num = (uint32_t)buf[i] * 127u + (sum_q15 >> 1);
        uint32_t val = num / sum_q15;
        y[i] = sat8_pos(val);
    }

    int32_t corr = 127;
    for (size_t i = 0; i < len; ++i) corr -= y[i];
    if (corr != 0) {
        int32_t v = (int32_t)y[len - 1] + corr;
        if (v < 0)   v = 0;
        if (v > 127) v = 127;
        y[len - 1] = (qint8_t)v;
    }

    if (buf != stack_buf) free(buf);
}
