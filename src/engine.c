// engine.c — TinyViTDet INT8 runtime (token-major), compatible with scalar kernels.
// Ready for future per-channel scalers (vectors) produced by onnx2int8.py.
//
// Notes:
//  • Uses scalar SC_* macros (max over per-channel) => drop-in with current kernels.
//  • Pre-loads pointers to per-channel Q0.15 vectors OFF_SC_* for later kernel upgrades.
//  • Head is applied PER TOKEN: out_head has shape [TOKENS, OUT_DIM] in int8.
//  • Input 'inp_patches' must be quantized in the PatchEmbed activation domain
//    consistent with packer (if you folded mean/std, pass raw patches; otherwise
//    pre-normalize in int-only before calling).
//
// MIT — 2025

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "kernel.h"   // matmul_int8, fused_qkv_int8, mha_int8, ffn_int8, layernorm_int8, sat8
#include "model.h"

typedef int8_t   qint8_t;
typedef int16_t  qint16_t;
typedef int32_t  qint32_t;

/* ---------------- Model dims / params ---------------- */
#define TOKENS      MODEL_TOKENS
#define LAYERS      MODEL_LAYERS
#define DMODEL      MODEL_DMODEL
#define DFF         MODEL_DFF
#define HEADS       MODEL_HEADS
#define PATCHES     MODEL_PATCHES
#define PATCH_DIM   MODEL_PATCH_DIM
#define OUT_DIM     MODEL_OUT_DIM
#define EPS_SHIFT   MODEL_EPS_SHIFT   /* kept for current LN kernels */
#define LN_EPS_Q31  LN_EPS_Q31        /* precise epsilon if your LN uses Q1.31 */

/* --------------- Weights access helpers --------------- */
extern const uint8_t weights_bin[];

#define PTR_I8(off)    ((const qint8_t  *)(weights_bin + (size_t)(off)))
#define PTR_I16(off)   ((const qint16_t *)(weights_bin + (size_t)(off)))
#define PTR_I32(off)   ((const qint32_t *)(weights_bin + (size_t)(off)))
#define PTR_OR_NULL_I16(off)  ((off) ? PTR_I16(off) : (const qint16_t*)NULL)

/* --------------------- Arena memory ------------------- */
__attribute__((aligned(16)))
static qint8_t arena_mem[ARENA_BYTES];

static qint8_t *bufA, *bufB;          /* ping-pong  TOKENS × DMODEL */
static qint8_t *q_buf, *k_buf, *v_buf;/* Q / K / V                  */
static qint8_t *tmp_ffn;              /* DFF × TOKENS workspace     */

/* ----------------- Top-level static ptrs -------------- */
static const qint8_t  *PE_W    = PTR_I8 (OFF_W_PE);
static const qint32_t *PE_B    = PTR_I32(OFF_B_PE);
static const qint8_t  *CLS_EMB = 
#if defined(OFF_CLS_EMB)
    PTR_I8 (OFF_CLS_EMB);
#else
    NULL;
#endif

static const qint8_t  *POS_EMB = 
#if defined(OFF_POS_EMB)
    PTR_I8 (OFF_POS_EMB);
#else
    NULL;
#endif

static const qint8_t  *HEAD_W  = PTR_I8 (OFF_HEAD_W);
static const qint32_t *HEAD_B  = PTR_I32(OFF_HEAD_B);

/* ---------------- Per-layer pointers ------------------ */
static const qint8_t  *W_QKV [LAYERS];
static const qint32_t *B_QKV [LAYERS];
static const qint8_t  *W_MHA [LAYERS];
static const qint32_t *B_MHA [LAYERS];
static const qint8_t  *W_FFN1[LAYERS];
static const qint32_t *B_FFN1[LAYERS];
static const qint8_t  *W_FFN2[LAYERS];
static const qint32_t *B_FFN2[LAYERS];

static const qint32_t *LN1_GAMMA[LAYERS];
static const qint32_t *LN1_BETA [LAYERS];
static const qint32_t *LN2_GAMMA[LAYERS];
static const qint32_t *LN2_BETA [LAYERS];

/* ---- Per-channel scaler vectors (Q0.15, optional now) ---- */
static const qint16_t *SCV_QKV [LAYERS];   /* len = SC_LEN_QKV   */
static const qint16_t *SCV_MHAO[LAYERS];   /* len = SC_LEN_MHA_O */
static const qint16_t *SCV_FFN1[LAYERS];   /* len = SC_LEN_FFN1  */
static const qint16_t *SCV_FFN2[LAYERS];   /* len = SC_LEN_FFN2  */
static const qint16_t *SCV_HEAD = NULL;    /* len = SC_LEN_HEAD  */

/* -------------- Legacy scalar SC_* (Q0.15) ------------- */
/* Designated init picks up only the indices defined in model.h */
static const qint16_t SC_QKV_scalar[] = {
#if defined(SC_QKV_L0)
    [0] = SC_QKV_L0,
#endif
#if defined(SC_QKV_L1)
    [1] = SC_QKV_L1,
#endif
#if defined(SC_QKV_L2)
    [2] = SC_QKV_L2,
#endif
#if defined(SC_QKV_L3)
    [3] = SC_QKV_L3,
#endif
#if defined(SC_QKV_L4)
    [4] = SC_QKV_L4,
#endif
#if defined(SC_QKV_L5)
    [5] = SC_QKV_L5,
#endif
#if defined(SC_QKV_L6)
    [6] = SC_QKV_L6,
#endif
#if defined(SC_QKV_L7)
    [7] = SC_QKV_L7,
#endif
#if defined(SC_QKV_L8)
    [8] = SC_QKV_L8,
#endif
#if defined(SC_QKV_L9)
    [9] = SC_QKV_L9,
#endif
#if defined(SC_QKV_L10)
    [10] = SC_QKV_L10,
#endif
#if defined(SC_QKV_L11)
    [11] = SC_QKV_L11,
#endif
};

static const qint16_t SC_MHAO_scalar[] = {
#if defined(SC_MHA_O_L0)
    [0] = SC_MHA_O_L0,
#endif
#if defined(SC_MHA_O_L1)
    [1] = SC_MHA_O_L1,
#endif
#if defined(SC_MHA_O_L2)
    [2] = SC_MHA_O_L2,
#endif
#if defined(SC_MHA_O_L3)
    [3] = SC_MHA_O_L3,
#endif
#if defined(SC_MHA_O_L4)
    [4] = SC_MHA_O_L4,
#endif
#if defined(SC_MHA_O_L5)
    [5] = SC_MHA_O_L5,
#endif
#if defined(SC_MHA_O_L6)
    [6] = SC_MHA_O_L6,
#endif
#if defined(SC_MHA_O_L7)
    [7] = SC_MHA_O_L7,
#endif
#if defined(SC_MHA_O_L8)
    [8] = SC_MHA_O_L8,
#endif
#if defined(SC_MHA_O_L9)
    [9] = SC_MHA_O_L9,
#endif
#if defined(SC_MHA_O_L10)
    [10] = SC_MHA_O_L10,
#endif
#if defined(SC_MHA_O_L11)
    [11] = SC_MHA_O_L11,
#endif
};

static const qint16_t SC_FFN1_scalar[] = {
#if defined(SC_FFN1_L0)
    [0] = SC_FFN1_L0,
#endif
#if defined(SC_FFN1_L1)
    [1] = SC_FFN1_L1,
#endif
#if defined(SC_FFN1_L2)
    [2] = SC_FFN1_L2,
#endif
#if defined(SC_FFN1_L3)
    [3] = SC_FFN1_L3,
#endif
#if defined(SC_FFN1_L4)
    [4] = SC_FFN1_L4,
#endif
#if defined(SC_FFN1_L5)
    [5] = SC_FFN1_L5,
#endif
#if defined(SC_FFN1_L6)
    [6] = SC_FFN1_L6,
#endif
#if defined(SC_FFN1_L7)
    [7] = SC_FFN1_L7,
#endif
#if defined(SC_FFN1_L8)
    [8] = SC_FFN1_L8,
#endif
#if defined(SC_FFN1_L9)
    [9] = SC_FFN1_L9,
#endif
#if defined(SC_FFN1_L10)
    [10] = SC_FFN1_L10,
#endif
#if defined(SC_FFN1_L11)
    [11] = SC_FFN1_L11,
#endif
};

static const qint16_t SC_FFN2_scalar[] = {
#if defined(SC_FFN2_L0)
    [0] = SC_FFN2_L0,
#endif
#if defined(SC_FFN2_L1)
    [1] = SC_FFN2_L1,
#endif
#if defined(SC_FFN2_L2)
    [2] = SC_FFN2_L2,
#endif
#if defined(SC_FFN2_L3)
    [3] = SC_FFN2_L3,
#endif
#if defined(SC_FFN2_L4)
    [4] = SC_FFN2_L4,
#endif
#if defined(SC_FFN2_L5)
    [5] = SC_FFN2_L5,
#endif
#if defined(SC_FFN2_L6)
    [6] = SC_FFN2_L6,
#endif
#if defined(SC_FFN2_L7)
    [7] = SC_FFN2_L7,
#endif
#if defined(SC_FFN2_L8)
    [8] = SC_FFN2_L8,
#endif
#if defined(SC_FFN2_L9)
    [9] = SC_FFN2_L9,
#endif
#if defined(SC_FFN2_L10)
    [10] = SC_FFN2_L10,
#endif
#if defined(SC_FFN2_L11)
    [11] = SC_FFN2_L11,
#endif
};

static inline qint16_t SC_HEAD_scalar(void) { return (qint16_t)SC_HEAD; }
static inline qint16_t SC_PE_scalar(void)   { return (qint16_t)SC_PE;   }

/* -------- OFF_* arrays using designated initializers -------- */
static const size_t OFF_W_QKV_arr[] = {
#if defined(OFF_W_QKV_L0)
    [0] = OFF_W_QKV_L0,
#endif
#if defined(OFF_W_QKV_L1)
    [1] = OFF_W_QKV_L1,
#endif
#if defined(OFF_W_QKV_L2)
    [2] = OFF_W_QKV_L2,
#endif
#if defined(OFF_W_QKV_L3)
    [3] = OFF_W_QKV_L3,
#endif
#if defined(OFF_W_QKV_L4)
    [4] = OFF_W_QKV_L4,
#endif
#if defined(OFF_W_QKV_L5)
    [5] = OFF_W_QKV_L5,
#endif
#if defined(OFF_W_QKV_L6)
    [6] = OFF_W_QKV_L6,
#endif
#if defined(OFF_W_QKV_L7)
    [7] = OFF_W_QKV_L7,
#endif
#if defined(OFF_W_QKV_L8)
    [8] = OFF_W_QKV_L8,
#endif
#if defined(OFF_W_QKV_L9)
    [9] = OFF_W_QKV_L9,
#endif
#if defined(OFF_W_QKV_L10)
    [10] = OFF_W_QKV_L10,
#endif
#if defined(OFF_W_QKV_L11)
    [11] = OFF_W_QKV_L11,
#endif
};

static const size_t OFF_B_QKV_arr[] = {
#if defined(OFF_B_QKV_L0)
    [0] = OFF_B_QKV_L0,
#endif
#if defined(OFF_B_QKV_L1)
    [1] = OFF_B_QKV_L1,
#endif
#if defined(OFF_B_QKV_L2)
    [2] = OFF_B_QKV_L2,
#endif
#if defined(OFF_B_QKV_L3)
    [3] = OFF_B_QKV_L3,
#endif
#if defined(OFF_B_QKV_L4)
    [4] = OFF_B_QKV_L4,
#endif
#if defined(OFF_B_QKV_L5)
    [5] = OFF_B_QKV_L5,
#endif
#if defined(OFF_B_QKV_L6)
    [6] = OFF_B_QKV_L6,
#endif
#if defined(OFF_B_QKV_L7)
    [7] = OFF_B_QKV_L7,
#endif
#if defined(OFF_B_QKV_L8)
    [8] = OFF_B_QKV_L8,
#endif
#if defined(OFF_B_QKV_L9)
    [9] = OFF_B_QKV_L9,
#endif
#if defined(OFF_B_QKV_L10)
    [10] = OFF_B_QKV_L10,
#endif
#if defined(OFF_B_QKV_L11)
    [11] = OFF_B_QKV_L11,
#endif
};

static const size_t OFF_W_O_arr[] = {
#if defined(OFF_W_O_L0)
    [0] = OFF_W_O_L0,
#endif
#if defined(OFF_W_O_L1)
    [1] = OFF_W_O_L1,
#endif
#if defined(OFF_W_O_L2)
    [2] = OFF_W_O_L2,
#endif
#if defined(OFF_W_O_L3)
    [3] = OFF_W_O_L3,
#endif
#if defined(OFF_W_O_L4)
    [4] = OFF_W_O_L4,
#endif
#if defined(OFF_W_O_L5)
    [5] = OFF_W_O_L5,
#endif
#if defined(OFF_W_O_L6)
    [6] = OFF_W_O_L6,
#endif
#if defined(OFF_W_O_L7)
    [7] = OFF_W_O_L7,
#endif
#if defined(OFF_W_O_L8)
    [8] = OFF_W_O_L8,
#endif
#if defined(OFF_W_O_L9)
    [9] = OFF_W_O_L9,
#endif
#if defined(OFF_W_O_L10)
    [10] = OFF_W_O_L10,
#endif
#if defined(OFF_W_O_L11)
    [11] = OFF_W_O_L11,
#endif
};

static const size_t OFF_B_O_arr[] = {
#if defined(OFF_B_O_L0)
    [0] = OFF_B_O_L0,
#endif
#if defined(OFF_B_O_L1)
    [1] = OFF_B_O_L1,
#endif
#if defined(OFF_B_O_L2)
    [2] = OFF_B_O_L2,
#endif
#if defined(OFF_B_O_L3)
    [3] = OFF_B_O_L3,
#endif
#if defined(OFF_B_O_L4)
    [4] = OFF_B_O_L4,
#endif
#if defined(OFF_B_O_L5)
    [5] = OFF_B_O_L5,
#endif
#if defined(OFF_B_O_L6)
    [6] = OFF_B_O_L6,
#endif
#if defined(OFF_B_O_L7)
    [7] = OFF_B_O_L7,
#endif
#if defined(OFF_B_O_L8)
    [8] = OFF_B_O_L8,
#endif
#if defined(OFF_B_O_L9)
    [9] = OFF_B_O_L9,
#endif
#if defined(OFF_B_O_L10)
    [10] = OFF_B_O_L10,
#endif
#if defined(OFF_B_O_L11)
    [11] = OFF_B_O_L11,
#endif
};

static const size_t OFF_W_FFN1_arr[] = {
#if defined(OFF_W_FFN1_L0)
    [0] = OFF_W_FFN1_L0,
#endif
#if defined(OFF_W_FFN1_L1)
    [1] = OFF_W_FFN1_L1,
#endif
#if defined(OFF_W_FFN1_L2)
    [2] = OFF_W_FFN1_L2,
#endif
#if defined(OFF_W_FFN1_L3)
    [3] = OFF_W_FFN1_L3,
#endif
#if defined(OFF_W_FFN1_L4)
    [4] = OFF_W_FFN1_L4,
#endif
#if defined(OFF_W_FFN1_L5)
    [5] = OFF_W_FFN1_L5,
#endif
#if defined(OFF_W_FFN1_L6)
    [6] = OFF_W_FFN1_L6,
#endif
#if defined(OFF_W_FFN1_L7)
    [7] = OFF_W_FFN1_L7,
#endif
#if defined(OFF_W_FFN1_L8)
    [8] = OFF_W_FFN1_L8,
#endif
#if defined(OFF_W_FFN1_L9)
    [9] = OFF_W_FFN1_L9,
#endif
#if defined(OFF_W_FFN1_L10)
    [10] = OFF_W_FFN1_L10,
#endif
#if defined(OFF_W_FFN1_L11)
    [11] = OFF_W_FFN1_L11,
#endif
};

static const size_t OFF_B_FFN1_arr[] = {
#if defined(OFF_B_FFN1_L0)
    [0] = OFF_B_FFN1_L0,
#endif
#if defined(OFF_B_FFN1_L1)
    [1] = OFF_B_FFN1_L1,
#endif
#if defined(OFF_B_FFN1_L2)
    [2] = OFF_B_FFN1_L2,
#endif
#if defined(OFF_B_FFN1_L3)
    [3] = OFF_B_FFN1_L3,
#endif
#if defined(OFF_B_FFN1_L4)
    [4] = OFF_B_FFN1_L4,
#endif
#if defined(OFF_B_FFN1_L5)
    [5] = OFF_B_FFN1_L5,
#endif
#if defined(OFF_B_FFN1_L6)
    [6] = OFF_B_FFN1_L6,
#endif
#if defined(OFF_B_FFN1_L7)
    [7] = OFF_B_FFN1_L7,
#endif
#if defined(OFF_B_FFN1_L8)
    [8] = OFF_B_FFN1_L8,
#endif
#if defined(OFF_B_FFN1_L9)
    [9] = OFF_B_FFN1_L9,
#endif
#if defined(OFF_B_FFN1_L10)
    [10] = OFF_B_FFN1_L10,
#endif
#if defined(OFF_B_FFN1_L11)
    [11] = OFF_B_FFN1_L11,
#endif
};

static const size_t OFF_W_FFN2_arr[] = {
#if defined(OFF_W_FFN2_L0)
    [0] = OFF_W_FFN2_L0,
#endif
#if defined(OFF_W_FFN2_L1)
    [1] = OFF_W_FFN2_L1,
#endif
#if defined(OFF_W_FFN2_L2)
    [2] = OFF_W_FFN2_L2,
#endif
#if defined(OFF_W_FFN2_L3)
    [3] = OFF_W_FFN2_L3,
#endif
#if defined(OFF_W_FFN2_L4)
    [4] = OFF_W_FFN2_L4,
#endif
#if defined(OFF_W_FFN2_L5)
    [5] = OFF_W_FFN2_L5,
#endif
#if defined(OFF_W_FFN2_L6)
    [6] = OFF_W_FFN2_L6,
#endif
#if defined(OFF_W_FFN2_L7)
    [7] = OFF_W_FFN2_L7,
#endif
#if defined(OFF_W_FFN2_L8)
    [8] = OFF_W_FFN2_L8,
#endif
#if defined(OFF_W_FFN2_L9)
    [9] = OFF_W_FFN2_L9,
#endif
#if defined(OFF_W_FFN2_L10)
    [10] = OFF_W_FFN2_L10,
#endif
#if defined(OFF_W_FFN2_L11)
    [11] = OFF_W_FFN2_L11,
#endif
};

static const size_t OFF_B_FFN2_arr[] = {
#if defined(OFF_B_FFN2_L0)
    [0] = OFF_B_FFN2_L0,
#endif
#if defined(OFF_B_FFN2_L1)
    [1] = OFF_B_FFN2_L1,
#endif
#if defined(OFF_B_FFN2_L2)
    [2] = OFF_B_FFN2_L2,
#endif
#if defined(OFF_B_FFN2_L3)
    [3] = OFF_B_FFN2_L3,
#endif
#if defined(OFF_B_FFN2_L4)
    [4] = OFF_B_FFN2_L4,
#endif
#if defined(OFF_B_FFN2_L5)
    [5] = OFF_B_FFN2_L5,
#endif
#if defined(OFF_B_FFN2_L6)
    [6] = OFF_B_FFN2_L6,
#endif
#if defined(OFF_B_FFN2_L7)
    [7] = OFF_B_FFN2_L7,
#endif
#if defined(OFF_B_FFN2_L8)
    [8] = OFF_B_FFN2_L8,
#endif
#if defined(OFF_B_FFN2_L9)
    [9] = OFF_B_FFN2_L9,
#endif
#if defined(OFF_B_FFN2_L10)
    [10] = OFF_B_FFN2_L10,
#endif
#if defined(OFF_B_FFN2_L11)
    [11] = OFF_B_FFN2_L11,
#endif
};

static const size_t OFF_G_LN1_arr[] = {
#if defined(OFF_G_LN1_L0)
    [0] = OFF_G_LN1_L0,
#endif
#if defined(OFF_G_LN1_L1)
    [1] = OFF_G_LN1_L1,
#endif
#if defined(OFF_G_LN1_L2)
    [2] = OFF_G_LN1_L2,
#endif
#if defined(OFF_G_LN1_L3)
    [3] = OFF_G_LN1_L3,
#endif
#if defined(OFF_G_LN1_L4)
    [4] = OFF_G_LN1_L4,
#endif
#if defined(OFF_G_LN1_L5)
    [5] = OFF_G_LN1_L5,
#endif
#if defined(OFF_G_LN1_L6)
    [6] = OFF_G_LN1_L6,
#endif
#if defined(OFF_G_LN1_L7)
    [7] = OFF_G_LN1_L7,
#endif
#if defined(OFF_G_LN1_L8)
    [8] = OFF_G_LN1_L8,
#endif
#if defined(OFF_G_LN1_L9)
    [9] = OFF_G_LN1_L9,
#endif
#if defined(OFF_G_LN1_L10)
    [10] = OFF_G_LN1_L10,
#endif
#if defined(OFF_G_LN1_L11)
    [11] = OFF_G_LN1_L11,
#endif
};

static const size_t OFF_B_LN1_arr[] = {
#if defined(OFF_B_LN1_L0)
    [0] = OFF_B_LN1_L0,
#endif
#if defined(OFF_B_LN1_L1)
    [1] = OFF_B_LN1_L1,
#endif
#if defined(OFF_B_LN1_L2)
    [2] = OFF_B_LN1_L2,
#endif
#if defined(OFF_B_LN1_L3)
    [3] = OFF_B_LN1_L3,
#endif
#if defined(OFF_B_LN1_L4)
    [4] = OFF_B_LN1_L4,
#endif
#if defined(OFF_B_LN1_L5)
    [5] = OFF_B_LN1_L5,
#endif
#if defined(OFF_B_LN1_L6)
    [6] = OFF_B_LN1_L6,
#endif
#if defined(OFF_B_LN1_L7)
    [7] = OFF_B_LN1_L7,
#endif
#if defined(OFF_B_LN1_L8)
    [8] = OFF_B_LN1_L8,
#endif
#if defined(OFF_B_LN1_L9)
    [9] = OFF_B_LN1_L9,
#endif
#if defined(OFF_B_LN1_L10)
    [10] = OFF_B_LN1_L10,
#endif
#if defined(OFF_B_LN1_L11)
    [11] = OFF_B_LN1_L11,
#endif
};

static const size_t OFF_G_LN2_arr[] = {
#if defined(OFF_G_LN2_L0)
    [0] = OFF_G_LN2_L0,
#endif
#if defined(OFF_G_LN2_L1)
    [1] = OFF_G_LN2_L1,
#endif
#if defined(OFF_G_LN2_L2)
    [2] = OFF_G_LN2_L2,
#endif
#if defined(OFF_G_LN2_L3)
    [3] = OFF_G_LN2_L3,
#endif
#if defined(OFF_G_LN2_L4)
    [4] = OFF_G_LN2_L4,
#endif
#if defined(OFF_G_LN2_L5)
    [5] = OFF_G_LN2_L5,
#endif
#if defined(OFF_G_LN2_L6)
    [6] = OFF_G_LN2_L6,
#endif
#if defined(OFF_G_LN2_L7)
    [7] = OFF_G_LN2_L7,
#endif
#if defined(OFF_G_LN2_L8)
    [8] = OFF_G_LN2_L8,
#endif
#if defined(OFF_G_LN2_L9)
    [9] = OFF_G_LN2_L9,
#endif
#if defined(OFF_G_LN2_L10)
    [10] = OFF_G_LN2_L10,
#endif
#if defined(OFF_G_LN2_L11)
    [11] = OFF_G_LN2_L11,
#endif
};

static const size_t OFF_B_LN2_arr[] = {
#if defined(OFF_B_LN2_L0)
    [0] = OFF_B_LN2_L0,
#endif
#if defined(OFF_B_LN2_L1)
    [1] = OFF_B_LN2_L1,
#endif
#if defined(OFF_B_LN2_L2)
    [2] = OFF_B_LN2_L2,
#endif
#if defined(OFF_B_LN2_L3)
    [3] = OFF_B_LN2_L3,
#endif
#if defined(OFF_B_LN2_L4)
    [4] = OFF_B_LN2_L4,
#endif
#if defined(OFF_B_LN2_L5)
    [5] = OFF_B_LN2_L5,
#endif
#if defined(OFF_B_LN2_L6)
    [6] = OFF_B_LN2_L6,
#endif
#if defined(OFF_B_LN2_L7)
    [7] = OFF_B_LN2_L7,
#endif
#if defined(OFF_B_LN2_L8)
    [8] = OFF_B_LN2_L8,
#endif
#if defined(OFF_B_LN2_L9)
    [9] = OFF_B_LN2_L9,
#endif
#if defined(OFF_B_LN2_L10)
    [10] = OFF_B_LN2_L10,
#endif
#if defined(OFF_B_LN2_L11)
    [11] = OFF_B_LN2_L11,
#endif
};

/* --------- OFF_SC_* per-layer vectors (optional) --------- */
static const size_t OFF_SC_QKV_arr[] = {
#if defined(OFF_SC_QKV_L0)
    [0] = OFF_SC_QKV_L0,
#endif
#if defined(OFF_SC_QKV_L1)
    [1] = OFF_SC_QKV_L1,
#endif
#if defined(OFF_SC_QKV_L2)
    [2] = OFF_SC_QKV_L2,
#endif
#if defined(OFF_SC_QKV_L3)
    [3] = OFF_SC_QKV_L3,
#endif
#if defined(OFF_SC_QKV_L4)
    [4] = OFF_SC_QKV_L4,
#endif
#if defined(OFF_SC_QKV_L5)
    [5] = OFF_SC_QKV_L5,
#endif
#if defined(OFF_SC_QKV_L6)
    [6] = OFF_SC_QKV_L6,
#endif
#if defined(OFF_SC_QKV_L7)
    [7] = OFF_SC_QKV_L7,
#endif
#if defined(OFF_SC_QKV_L8)
    [8] = OFF_SC_QKV_L8,
#endif
#if defined(OFF_SC_QKV_L9)
    [9] = OFF_SC_QKV_L9,
#endif
#if defined(OFF_SC_QKV_L10)
    [10] = OFF_SC_QKV_L10,
#endif
#if defined(OFF_SC_QKV_L11)
    [11] = OFF_SC_QKV_L11,
#endif
};

static const size_t OFF_SC_MHAO_arr[] = {
#if defined(OFF_SC_MHA_O_L0)
    [0] = OFF_SC_MHA_O_L0,
#endif
#if defined(OFF_SC_MHA_O_L1)
    [1] = OFF_SC_MHA_O_L1,
#endif
#if defined(OFF_SC_MHA_O_L2)
    [2] = OFF_SC_MHA_O_L2,
#endif
#if defined(OFF_SC_MHA_O_L3)
    [3] = OFF_SC_MHA_O_L3,
#endif
#if defined(OFF_SC_MHA_O_L4)
    [4] = OFF_SC_MHA_O_L4,
#endif
#if defined(OFF_SC_MHA_O_L5)
    [5] = OFF_SC_MHA_O_L5,
#endif
#if defined(OFF_SC_MHA_O_L6)
    [6] = OFF_SC_MHA_O_L6,
#endif
#if defined(OFF_SC_MHA_O_L7)
    [7] = OFF_SC_MHA_O_L7,
#endif
#if defined(OFF_SC_MHA_O_L8)
    [8] = OFF_SC_MHA_O_L8,
#endif
#if defined(OFF_SC_MHA_O_L9)
    [9] = OFF_SC_MHA_O_L9,
#endif
#if defined(OFF_SC_MHA_O_L10)
    [10] = OFF_SC_MHA_O_L10,
#endif
#if defined(OFF_SC_MHA_O_L11)
    [11] = OFF_SC_MHA_O_L11,
#endif
};

static const size_t OFF_SC_FFN1_arr[] = {
#if defined(OFF_SC_FFN1_L0)
    [0] = OFF_SC_FFN1_L0,
#endif
#if defined(OFF_SC_FFN1_L1)
    [1] = OFF_SC_FFN1_L1,
#endif
#if defined(OFF_SC_FFN1_L2)
    [2] = OFF_SC_FFN1_L2,
#endif
#if defined(OFF_SC_FFN1_L3)
    [3] = OFF_SC_FFN1_L3,
#endif
#if defined(OFF_SC_FFN1_L4)
    [4] = OFF_SC_FFN1_L4,
#endif
#if defined(OFF_SC_FFN1_L5)
    [5] = OFF_SC_FFN1_L5,
#endif
#if defined(OFF_SC_FFN1_L6)
    [6] = OFF_SC_FFN1_L6,
#endif
#if defined(OFF_SC_FFN1_L7)
    [7] = OFF_SC_FFN1_L7,
#endif
#if defined(OFF_SC_FFN1_L8)
    [8] = OFF_SC_FFN1_L8,
#endif
#if defined(OFF_SC_FFN1_L9)
    [9] = OFF_SC_FFN1_L9,
#endif
#if defined(OFF_SC_FFN1_L10)
    [10] = OFF_SC_FFN1_L10,
#endif
#if defined(OFF_SC_FFN1_L11)
    [11] = OFF_SC_FFN1_L11,
#endif
};

static const size_t OFF_SC_FFN2_arr[] = {
#if defined(OFF_SC_FFN2_L0)
    [0] = OFF_SC_FFN2_L0,
#endif
#if defined(OFF_SC_FFN2_L1)
    [1] = OFF_SC_FFN2_L1,
#endif
#if defined(OFF_SC_FFN2_L2)
    [2] = OFF_SC_FFN2_L2,
#endif
#if defined(OFF_SC_FFN2_L3)
    [3] = OFF_SC_FFN2_L3,
#endif
#if defined(OFF_SC_FFN2_L4)
    [4] = OFF_SC_FFN2_L4,
#endif
#if defined(OFF_SC_FFN2_L5)
    [5] = OFF_SC_FFN2_L5,
#endif
#if defined(OFF_SC_FFN2_L6)
    [6] = OFF_SC_FFN2_L6,
#endif
#if defined(OFF_SC_FFN2_L7)
    [7] = OFF_SC_FFN2_L7,
#endif
#if defined(OFF_SC_FFN2_L8)
    [8] = OFF_SC_FFN2_L8,
#endif
#if defined(OFF_SC_FFN2_L9)
    [9] = OFF_SC_FFN2_L9,
#endif
#if defined(OFF_SC_FFN2_L10)
    [10] = OFF_SC_FFN2_L10,
#endif
#if defined(OFF_SC_FFN2_L11)
    [11] = OFF_SC_FFN2_L11,
#endif
};

/* ---------------- Layernorm (token-major helper) ---------------- */
static inline void layernorm_tokens(qint8_t *x, const qint32_t *gamma, const qint32_t *beta)
{
    for (size_t t = 0; t < TOKENS; ++t) {
        layernorm_int8(x + t * DMODEL, gamma, beta, DMODEL, EPS_SHIFT);
        /* If/when you switch to Q1.31 epsilon:
           layernorm_int8_q31(x + t*DMODEL, gamma, beta, DMODEL, LN_EPS_Q31); */
    }
}

/* ---------------- PatchEmbed: write tokens [1..PATCHES], keep t=0 for [CLS] ---------------- */
static inline void patch_embed(const qint8_t *inp_patches)
{
    /* out: bufA + DMODEL → patch tokens in slots 1..PATCHES (token-major) */
    matmul_int8(inp_patches,
                PE_W, PE_B,
                bufA + DMODEL,
                PATCHES,           /* M (tokens without CLS) */
                DMODEL,            /* N */
                PATCH_DIM,         /* K */
                SC_PE_scalar());

    /* Copy [CLS] at t=0 if available */
    if (CLS_EMB) {
        memcpy(bufA, CLS_EMB, DMODEL);
    } else {
        /* Zero CLS if missing */
        memset(bufA, 0, DMODEL);
    }
}

/* ---------------- Add absolute positional embeddings (saturating add) ---------------- */
static inline void add_pos_emb(qint8_t *tokens)
{
    if (!POS_EMB) return;
    const size_t len = (size_t)TOKENS * (size_t)DMODEL;
    for (size_t i = 0; i < len; ++i) {
        tokens[i] = sat8((int32_t)tokens[i] + (int32_t)POS_EMB[i]);
    }
}

/* =============================== Transformer block =============================== */
static void transformer_block(size_t l, qint8_t *inp, qint8_t *out)
{
    const qint8_t  *Wqkv = W_QKV [l];
    const qint32_t *Bqkv = B_QKV [l];
    const qint8_t  *Wo   = W_MHA [l];
    const qint32_t *Bo   = B_MHA [l];
    const qint8_t  *W1   = W_FFN1[l];
    const qint32_t *B1   = B_FFN1[l];
    const qint8_t  *W2   = W_FFN2[l];
    const qint32_t *B2   = B_FFN2[l];

    const qint32_t *g1 = LN1_GAMMA[l];
    const qint32_t *b1 = LN1_BETA [l];
    const qint32_t *g2 = LN2_GAMMA[l];
    const qint32_t *b2 = LN2_BETA [l];

    const size_t vec_sz = (size_t)TOKENS * (size_t)DMODEL;

    qint8_t *skip1 = tmp_ffn;
    memcpy(skip1, inp, vec_sz);

    layernorm_tokens(inp, g1, b1);

    /* QKV projection (fused) — scalar scale for now */
    fused_qkv_int8(inp, Wqkv, Bqkv, q_buf, k_buf, v_buf, TOKENS, SC_QKV_scalar[l]);
    /* If/when kernels accept per-channel:
       fused_qkv_int8_pc(inp, Wqkv, Bqkv, q_buf, k_buf, v_buf, TOKENS,
                         SCV_QKV[l], SC_LEN_QKV); */

    /* Multi-head attention (online softmax assumed) */
    mha_int8(q_buf, k_buf, v_buf, out, TOKENS, HEADS, SC_QKV_scalar[l]);
    /* If/when per-channel for attention internals:
       mha_int8_pc(q_buf, k_buf, v_buf, out, TOKENS, HEADS, SCV_QKV[l], SC_LEN_QKV); */

    /* Output projection */
    matmul_int8(out, Wo, Bo, tmp_ffn, TOKENS, DMODEL, DMODEL, SC_MHAO_scalar[l]);
    memcpy(out, tmp_ffn, vec_sz);

    /* Residual add */
    for (size_t i = 0; i < vec_sz; ++i) out[i] = sat8((int32_t)skip1[i] + (int32_t)out[i]);

    /* FFN */
    qint8_t *skip2 = q_buf;
    memcpy(skip2, out, vec_sz);

    layernorm_tokens(out, g2, b2);

    /* ffn_int8 does: out = GELU(out*W1+B1) * W2 + B2 (int-only approx inside) */
    ffn_int8(out, W1, B1, W2, B2, tmp_ffn, out, TOKENS, SC_FFN1_scalar[l], SC_FFN2_scalar[l]);
    /* If/when per-channel:
       ffn_int8_pc(out, W1, B1, W2, B2, tmp_ffn, out, TOKENS,
                   SCV_FFN1[l], SC_LEN_FFN1, SCV_FFN2[l], SC_LEN_FFN2); */

    for (size_t i = 0; i < vec_sz; ++i) out[i] = sat8((int32_t)skip2[i] + (int32_t)out[i]);
}

/* ================================ Public API ================================= */

void vit_init(void)
{
    bufA    = arena_mem + ARENA_OFF_BUF0;
    bufB    = arena_mem + ARENA_OFF_BUF1;
    q_buf   = arena_mem + ARENA_OFF_Q;
    k_buf   = arena_mem + ARENA_OFF_K;
    v_buf   = arena_mem + ARENA_OFF_V;
    tmp_ffn = arena_mem + ARENA_OFF_TMP;

#ifdef DEBUG
    memset(arena_mem, 0, ARENA_BYTES);
#endif

    for (size_t l = 0; l < LAYERS; ++l) {
        W_QKV [l] = PTR_I8 (OFF_W_QKV_arr[l]);
        B_QKV [l] = PTR_I32(OFF_B_QKV_arr[l]);

        W_MHA [l] = PTR_I8 (OFF_W_O_arr   [l]);
        B_MHA [l] = PTR_I32(OFF_B_O_arr   [l]);

        W_FFN1[l] = PTR_I8 (OFF_W_FFN1_arr[l]);
        B_FFN1[l] = PTR_I32(OFF_B_FFN1_arr[l]);

        W_FFN2[l] = PTR_I8 (OFF_W_FFN2_arr[l]);
        B_FFN2[l] = PTR_I32(OFF_B_FFN2_arr[l]);

        LN1_GAMMA[l] = PTR_I32(OFF_G_LN1_arr[l]);
        LN1_BETA [l] = PTR_I32(OFF_B_LN1_arr[l]);
        LN2_GAMMA[l] = PTR_I32(OFF_G_LN2_arr[l]);
        LN2_BETA [l] = PTR_I32(OFF_B_LN2_arr[l]);

        /* Per-channel scalers (vectors) — may be NULL if header omitted them */
        const size_t off_sc_qkv  = (l < sizeof(OFF_SC_QKV_arr)/sizeof(OFF_SC_QKV_arr[0]))   ? OFF_SC_QKV_arr[l]  : 0;
        const size_t off_sc_mhao = (l < sizeof(OFF_SC_MHAO_arr)/sizeof(OFF_SC_MHAO_arr[0])) ? OFF_SC_MHAO_arr[l] : 0;
        const size_t off_sc_ffn1 = (l < sizeof(OFF_SC_FFN1_arr)/sizeof(OFF_SC_FFN1_arr[0])) ? OFF_SC_FFN1_arr[l] : 0;
        const size_t off_sc_ffn2 = (l < sizeof(OFF_SC_FFN2_arr)/sizeof(OFF_SC_FFN2_arr[0])) ? OFF_SC_FFN2_arr[l] : 0;

        SCV_QKV [l] = PTR_OR_NULL_I16(off_sc_qkv);
        SCV_MHAO[l] = PTR_OR_NULL_I16(off_sc_mhao);
        SCV_FFN1[l] = PTR_OR_NULL_I16(off_sc_ffn1);
        SCV_FFN2[l] = PTR_OR_NULL_I16(off_sc_ffn2);
    }

#if defined(OFF_SC_HEAD)
    SCV_HEAD = PTR_I16(OFF_SC_HEAD);
#else
    SCV_HEAD = NULL;
#endif
}

/* Forward
 *  inp_patches:  [PATCHES, PATCH_DIM] int8 (tokenized patches, no CLS)
 *  out_head:     [TOKENS,  OUT_DIM ] int8   (includes t=0 = CLS row)
 */
void vit_forward(const qint8_t *inp_patches,
                 qint8_t       *out_head)
{
    /* PatchEmbed + APE */
    patch_embed(inp_patches);
    add_pos_emb(bufA);

    qint8_t *cur  = bufA;
    qint8_t *next = bufB;

    for (size_t l = 0; l < LAYERS; ++l) {
        transformer_block(l, cur, next);
        qint8_t *tmp = cur; cur = next; next = tmp;
    }

    /* Head per-token (apply to ALL tokens, including CLS at t=0) */
    matmul_int8(cur, HEAD_W, HEAD_B, out_head, TOKENS, OUT_DIM, DMODEL, SC_HEAD_scalar());
    /* If/when per-channel head is supported:
       matmul_int8_pc(cur, HEAD_W, HEAD_B, out_head, TOKENS, OUT_DIM, DMODEL,
                      SCV_HEAD, SC_LEN_HEAD); */
}
