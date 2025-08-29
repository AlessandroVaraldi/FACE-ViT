// engine.h
#ifndef ENGINE_H_
#define ENGINE_H_

#include <stdint.h>
#include "kernel.h"
#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif


#define VIT_INPUT_SIZE  (MODEL_PATCHES * MODEL_PATCH_DIM)
#define VIT_OUTPUT_SIZE MODEL_OUT_DIM

void vit_init(void);

void vit_forward(const qint8_t *inp_patches, qint8_t *logits_out);

#ifdef __cplusplus
}
#endif

#endif // ENGINE_H_
