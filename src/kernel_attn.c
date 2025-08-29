#include "kernel.h"
#include "model.h"
#include <string.h>

void fused_qkv_int8(const qint8_t  *inp,
                    const qint8_t  *W_qkv,
                    const qint32_t *bias_qkv,
                    qint8_t        *q_out,
                    qint8_t        *k_out,
                    qint8_t        *v_out,
                    size_t          tokens,
                    int16_t         mult_q15)
{
    qint8_t tmp_row[3 * MODEL_DMODEL];

    for (size_t t = 0; t < tokens; ++t) {
        const qint8_t *inp_row = inp + t * MODEL_DMODEL;

        matmul_int8(inp_row,
                    W_qkv,
                    bias_qkv,
                    tmp_row,
                    1,
                    3 * MODEL_DMODEL,
                    MODEL_DMODEL,
                    mult_q15);

        memcpy(q_out + t * MODEL_DMODEL,                 tmp_row,                       MODEL_DMODEL);
        memcpy(k_out + t * MODEL_DMODEL,                 tmp_row + MODEL_DMODEL,        MODEL_DMODEL);
        memcpy(v_out + t * MODEL_DMODEL,                 tmp_row + 2 * MODEL_DMODEL,    MODEL_DMODEL);
    }
}

void mha_int8(const qint8_t *Q,
              const qint8_t *K,
              const qint8_t *V,
              qint8_t       *O,
              size_t         tokens,
              uint8_t        heads,
              int16_t        mult_q15)
{
    const size_t dim_head = MODEL_DMODEL / heads;

    qint8_t scores[tokens];
    qint8_t attn  [tokens];

    for (uint8_t h = 0; h < heads; ++h) {
        const size_t h_off = h * dim_head;
        for (size_t i = 0; i < tokens; ++i) {
            for (size_t j = 0; j < tokens; ++j) {
                int32_t acc = 0;
                const qint8_t *q_ptr = Q + i * MODEL_DMODEL + h_off;
                const qint8_t *k_ptr = K + j * MODEL_DMODEL + h_off;

                for (size_t d = 0; d < dim_head; ++d)
                    acc += (int32_t)q_ptr[d] * (int32_t)k_ptr[d];

                scores[j] = requant_q15(acc, mult_q15);
            }

            softmax_int8(scores, attn, tokens);

            qint8_t *o_ptr = O + i * MODEL_DMODEL + h_off;

            for (size_t d = 0; d < dim_head; ++d) {
                int32_t acc = 0;
                for (size_t j = 0; j < tokens; ++j) {
                    const qint8_t *v_ptr = V + j * MODEL_DMODEL + h_off;
                    acc += (int32_t)attn[j] * (int32_t)v_ptr[d];
                }
                o_ptr[d] = requant_q15(acc, mult_q15);
            }
        }
    }
}
