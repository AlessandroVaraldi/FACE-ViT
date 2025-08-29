#include "kernel.h"
#include "model.h"
#include <string.h>

void ffn_int8(const qint8_t  *inp,
              const qint8_t  *W1,
              const qint32_t *B1,
              const qint8_t  *W2,
              const qint32_t *B2,
              qint8_t        *tmp,
              qint8_t        *out,
              size_t          tokens,
              int16_t         mult1_q15,
              int16_t         mult2_q15)
{
    matmul_int8(inp,
                W1, B1,
                tmp,
                tokens,             /* M */
                MODEL_DFF,          /* N */
                MODEL_DMODEL,       /* K */
                mult1_q15);

    for (size_t t = 0; t < tokens; ++t) {
        qint8_t *row = tmp + t * MODEL_DFF;
        gelu_int8(row, MODEL_DFF);
    }

    matmul_int8(tmp,
                W2, B2,
                out,
                tokens,             /* M */
                MODEL_DMODEL,       /* N */
                MODEL_DFF,          /* K */
                mult2_q15);
}
