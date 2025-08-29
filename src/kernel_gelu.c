#include "kernel.h"
#include "model.h"
#include <stdint.h>

#define Q15(x)      ((int32_t)((x) * 32768 + ((x) >= 0 ? 0.5f : -0.5f)))

static const int32_t A_Q15        = Q15(-0.2888f);   /* −9463  */
static const int32_t B_Q15        = Q15(-1.769f);    /* −57917 */
static const int32_t NEG_B_ABS_Q15= 57917;           /* |−B|    */
static const int32_t ONE_Q15      = 32768;           /* 1.0     */
static const int32_t INV_SQRT2_Q15= Q15(0.70710678f);/* 1/√2    */

void gelu_int8(qint8_t *x, size_t len)
{
    for (size_t i = 0; i < len; ++i) {

        int32_t x_q15 = ((int32_t)x[i]) << 15;
        int32_t u_q15 = (x_q15 * INV_SQRT2_Q15) >> 15;

        int32_t abs_u = u_q15 >= 0 ? u_q15 : -u_q15;
        if (abs_u > NEG_B_ABS_Q15) abs_u = NEG_B_ABS_Q15;

        int32_t t_q15 = abs_u + B_Q15;
        int64_t t_sq  = (int64_t)t_q15 * t_q15;
        int32_t t_sq_q15 = (int32_t)(t_sq >> 15);
        int32_t s_q15 = (int32_t)(((int64_t)A_Q15 * t_sq_q15) >> 15);
        s_q15 += ONE_Q15;

        if (u_q15 < 0) s_q15 = -s_q15;

        int32_t one_plus_L = ONE_Q15 + s_q15;
        int32_t prod       = ((int32_t)x[i]) * one_plus_L;
        int32_t y_q15      = (prod >> 1);

        int32_t y = (y_q15 + (1 << 14)) >> 15;
        x[i] = sat8(y);
    }
}
