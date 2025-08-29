#ifndef KERNEL_H
#define KERNEL_H 1

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h> 

typedef int8_t  qint8_t;
typedef int32_t qint32_t;

static inline qint8_t sat8(int32_t x)
{
    if (x > 127)  return 127;
    if (x < -128) return -128;
    return (qint8_t)x;
}


static inline qint8_t requant_q15(int32_t acc, int16_t mult_q15)
{
    int32_t tmp = (int32_t)(((int64_t)acc * mult_q15 + (1 << 14)) >> 15);
    return sat8(tmp);
}


void matmul_int8(const qint8_t  *A,
                 const qint8_t  *B,
                 const qint32_t *bias,
                 qint8_t        *C,
                 size_t          M,
                 size_t          N,
                 size_t          K,
                 int16_t         mult_q15);

void fused_qkv_int8(const qint8_t  *inp,
                    const qint8_t  *W_qkv,
                    const qint32_t *bias_qkv,
                    qint8_t        *q_out,
                    qint8_t        *k_out,
                    qint8_t        *v_out,
                    size_t          tokens,
                    int16_t         mult_q15);

void mha_int8(const qint8_t *Q,
              const qint8_t *K,
              const qint8_t *V,
              qint8_t       *O,
              size_t         tokens,
              uint8_t        heads,
              int16_t        mult_q15);

void ffn_int8(const qint8_t  *inp,
              const qint8_t  *W1,
              const qint32_t *b1,
              const qint8_t  *W2,
              const qint32_t *b2,
              qint8_t        *tmp,
              qint8_t        *out,
              size_t          tokens,
              int16_t         mult1_q15,
              int16_t         mult2_q15);

void gelu_int8       (qint8_t *x, size_t len);
void softmax_int8    (const qint8_t *x, qint8_t *y, size_t len);
void layernorm_int8  (qint8_t        *x,
                      const qint32_t *gamma,
                      const qint32_t *beta,
                      size_t          len,
                      uint8_t         epsilon_shift);

#ifdef __cplusplus
}
#endif
#endif /* KERNEL_H */
