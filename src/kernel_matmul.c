#include "kernel.h"
#include "model.h"
#include <string.h>
#include <stdlib.h>

void matmul_int8 (const qint8_t  *A,
                  const qint8_t  *B,
                  const qint32_t *bias,
                  qint8_t        *C,
                  size_t          M,
                  size_t          N,
                  size_t          K,
                  int16_t         mult_q15)
{
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {

            int32_t acc = bias ? bias[j] : 0;

            const qint8_t *a_ptr = A + i * K;
            const qint8_t *b_ptr = B + j;

            for (size_t k = 0; k < K; ++k)
                acc += (int32_t)a_ptr[k] * (int32_t)b_ptr[k * N];

            C[i * N + j] = requant_q15(acc, mult_q15); 
        }
    }
}
