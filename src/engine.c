#include <stdint.h>
#include <string.h>

#include "kernel.h"
#include "model.h"

typedef int8_t  qint8_t;
typedef int32_t qint32_t;

#define TOKENS      MODEL_TOKENS
#define LAYERS      MODEL_LAYERS
#define DMODEL      MODEL_DMODEL
#define DFF         MODEL_DFF
#define HEADS       MODEL_HEADS
#define PATCHES     MODEL_PATCHES
#define PATCH_DIM   MODEL_PATCH_DIM
#define OUT_DIM     MODEL_OUT_DIM
#define EPS_SHIFT   MODEL_EPS_SHIFT

#define PTR_I8(off)   ((const qint8_t  *)(weights_bin + (off)))
#define PTR_I32(off)  ((const qint32_t *)(weights_bin + (off)))

__attribute__((aligned(16)))
static qint8_t arena_mem[ARENA_BYTES];

static qint8_t *bufA, *bufB;          /* ping-pong  TOKENS × DMODEL */
static qint8_t *q_buf, *k_buf, *v_buf;/* Q / K / V                  */
static qint8_t *tmp_ffn;              /* DFF × TOKENS workspace     */

static const qint8_t  *PE_W    = PTR_I8 (OFF_W_PE);
static const qint32_t *PE_B    = PTR_I32(OFF_B_PE);
static const qint8_t  *CLS_EMB = PTR_I8 (OFF_CLS_EMB);
static const qint8_t  *POS_EMB = PTR_I8 (OFF_POS_EMB);

static const qint8_t  *HEAD_W  = PTR_I8 (OFF_HEAD_W);
static const qint32_t *HEAD_B  = PTR_I32(OFF_HEAD_B);

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

static const size_t OFF_W_QKV_[LAYERS] = {
    OFF_W_QKV_L0, OFF_W_QKV_L1, OFF_W_QKV_L2, OFF_W_QKV_L3,
    OFF_W_QKV_L4, OFF_W_QKV_L5, OFF_W_QKV_L6, OFF_W_QKV_L7 };

static const size_t OFF_B_QKV_[LAYERS] = {
    OFF_B_QKV_L0, OFF_B_QKV_L1, OFF_B_QKV_L2, OFF_B_QKV_L3,
    OFF_B_QKV_L4, OFF_B_QKV_L5, OFF_B_QKV_L6, OFF_B_QKV_L7 };

static const size_t OFF_W_O_[LAYERS] = {
    OFF_W_O_L0, OFF_W_O_L1, OFF_W_O_L2, OFF_W_O_L3,
    OFF_W_O_L4, OFF_W_O_L5, OFF_W_O_L6, OFF_W_O_L7 };

static const size_t OFF_B_O_[LAYERS] = {
    OFF_B_O_L0, OFF_B_O_L1, OFF_B_O_L2, OFF_B_O_L3,
    OFF_B_O_L4, OFF_B_O_L5, OFF_B_O_L6, OFF_B_O_L7 };

static const size_t OFF_W_FFN1_[LAYERS] = {
    OFF_W_FFN1_L0, OFF_W_FFN1_L1, OFF_W_FFN1_L2, OFF_W_FFN1_L3,
    OFF_W_FFN1_L4, OFF_W_FFN1_L5, OFF_W_FFN1_L6, OFF_W_FFN1_L7 };

static const size_t OFF_B_FFN1_[LAYERS] = {
    OFF_B_FFN1_L0, OFF_B_FFN1_L1, OFF_B_FFN1_L2, OFF_B_FFN1_L3,
    OFF_B_FFN1_L4, OFF_B_FFN1_L5, OFF_B_FFN1_L6, OFF_B_FFN1_L7 };

static const size_t OFF_W_FFN2_[LAYERS] = {
    OFF_W_FFN2_L0, OFF_W_FFN2_L1, OFF_W_FFN2_L2, OFF_W_FFN2_L3,
    OFF_W_FFN2_L4, OFF_W_FFN2_L5, OFF_W_FFN2_L6, OFF_W_FFN2_L7 };

static const size_t OFF_B_FFN2_[LAYERS] = {
    OFF_B_FFN2_L0, OFF_B_FFN2_L1, OFF_B_FFN2_L2, OFF_B_FFN2_L3,
    OFF_B_FFN2_L4, OFF_B_FFN2_L5, OFF_B_FFN2_L6, OFF_B_FFN2_L7 };

static const size_t OFF_G_LN1_[LAYERS] = {
    OFF_G_LN1_L0, OFF_G_LN1_L1, OFF_G_LN1_L2, OFF_G_LN1_L3,
    OFF_G_LN1_L4, OFF_G_LN1_L5, OFF_G_LN1_L6, OFF_G_LN1_L7 };

static const size_t OFF_B_LN1_[LAYERS] = {
    OFF_B_LN1_L0, OFF_B_LN1_L1, OFF_B_LN1_L2, OFF_B_LN1_L3,
    OFF_B_LN1_L4, OFF_B_LN1_L5, OFF_B_LN1_L6, OFF_B_LN1_L7 };

static const size_t OFF_G_LN2_[LAYERS] = {
    OFF_G_LN2_L0, OFF_G_LN2_L1, OFF_G_LN2_L2, OFF_G_LN2_L3,
    OFF_G_LN2_L4, OFF_G_LN2_L5, OFF_G_LN2_L6, OFF_G_LN2_L7 };

static const size_t OFF_B_LN2_[LAYERS] = {
    OFF_B_LN2_L0, OFF_B_LN2_L1, OFF_B_LN2_L2, OFF_B_LN2_L3,
    OFF_B_LN2_L4, OFF_B_LN2_L5, OFF_B_LN2_L6, OFF_B_LN2_L7 };

static const int16_t SC_PE_q15   = SC_PE;
static const int16_t SC_HEAD_q15 = SC_HEAD;

static const int16_t SC_QKV_q15[LAYERS]  = {
    SC_QKV_L0,  SC_QKV_L1,  SC_QKV_L2,  SC_QKV_L3,
    SC_QKV_L4,  SC_QKV_L5,  SC_QKV_L6,  SC_QKV_L7 };

static const int16_t SC_MHA_O_q15[LAYERS] = {
    SC_MHA_O_L0, SC_MHA_O_L1, SC_MHA_O_L2, SC_MHA_O_L3,
    SC_MHA_O_L4, SC_MHA_O_L5, SC_MHA_O_L6, SC_MHA_O_L7 };

static const int16_t SC_FFN1_q15[LAYERS] = {
    SC_FFN1_L0, SC_FFN1_L1, SC_FFN1_L2, SC_FFN1_L3,
    SC_FFN1_L4, SC_FFN1_L5, SC_FFN1_L6, SC_FFN1_L7 };

static const int16_t SC_FFN2_q15[LAYERS] = {
    SC_FFN2_L0, SC_FFN2_L1, SC_FFN2_L2, SC_FFN2_L3,
    SC_FFN2_L4, SC_FFN2_L5, SC_FFN2_L6, SC_FFN2_L7 };

static inline void layernorm_tokens(qint8_t *x, const qint32_t *gamma, const qint32_t *beta)
{
    for (size_t t = 0; t < TOKENS; ++t) layernorm_int8(x + t * DMODEL, gamma, beta, DMODEL, EPS_SHIFT);
}

/* Scrive i patch-token a partire da t=1 e riserva t=0 per [CLS] */
void patch_embed(const qint8_t *inp_patches)
{
    /* Output a partire da bufA + DMODEL → i patch vanno in [1..PATCHES] */
    matmul_int8(inp_patches,
                PE_W, PE_B,
                bufA + DMODEL,
                PATCHES,           /* M */
                DMODEL,            /* N */
                PATCH_DIM,         /* K */
                SC_PE_q15);

    /* Copia [CLS] in testa (t=0) */
    memcpy(bufA, CLS_EMB, DMODEL);
}

/* Somma APE per-token (saturazione int8) */
static inline void add_pos_emb(qint8_t *tokens)
{
    const size_t len = (size_t)TOKENS * (size_t)DMODEL;
    for (size_t i = 0; i < len; ++i) {
        tokens[i] = sat8((int32_t)tokens[i] + (int32_t)POS_EMB[i]);
    }
}

/* ============================================================= */
/*  Transformer block                                            */
/* ============================================================= */
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

    const size_t vec_sz = TOKENS * DMODEL;

    qint8_t *skip1 = tmp_ffn;
    memcpy(skip1, inp, vec_sz);

    layernorm_tokens(inp, g1, b1);

    fused_qkv_int8(inp, Wqkv, Bqkv, q_buf, k_buf, v_buf, TOKENS, SC_QKV_q15[l]);

    mha_int8(q_buf, k_buf, v_buf, out, TOKENS, HEADS, SC_QKV_q15[l]);

    matmul_int8(out, Wo, Bo, tmp_ffn, TOKENS, DMODEL, DMODEL, SC_MHA_O_q15[l]);
    memcpy(out, tmp_ffn, TOKENS * DMODEL);

    for (size_t i = 0; i < vec_sz; ++i) out[i] = sat8((int32_t)skip1[i] + out[i]);

    qint8_t *skip2 = q_buf;
    memcpy(skip2, out, vec_sz);

    layernorm_tokens(out, g2, b2);

    ffn_int8(out, W1, B1, W2, B2, tmp_ffn, out, TOKENS, SC_FFN1_q15[l], SC_FFN2_q15[l]);

    for (size_t i = 0; i < vec_sz; ++i) out[i] = sat8((int32_t)skip2[i] + out[i]);
}

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
        W_QKV [l] = PTR_I8 (OFF_W_QKV_ [l]);
        B_QKV [l] = PTR_I32(OFF_B_QKV_ [l]);

        W_MHA [l] = PTR_I8 (OFF_W_O_   [l]);
        B_MHA [l] = PTR_I32(OFF_B_O_   [l]);

        W_FFN1[l] = PTR_I8 (OFF_W_FFN1_[l]);
        B_FFN1[l] = PTR_I32(OFF_B_FFN1_[l]);

        W_FFN2[l] = PTR_I8 (OFF_W_FFN2_[l]);
        B_FFN2[l] = PTR_I32(OFF_B_FFN2_[l]);

        LN1_GAMMA[l] = PTR_I32(OFF_G_LN1_[l]);
        LN1_BETA [l] = PTR_I32(OFF_B_LN1_[l]);
        LN2_GAMMA[l] = PTR_I32(OFF_G_LN2_[l]);
        LN2_BETA [l] = PTR_I32(OFF_B_LN2_[l]);
    }
}

void vit_forward(const qint8_t *inp_patches,
                 qint8_t       *logits_out)
{
    patch_embed(inp_patches);
    add_pos_emb(bufA);

    qint8_t *cur  = bufA;
    qint8_t *next = bufB;

    for (size_t l = 0; l < LAYERS; ++l) {
        transformer_block(l, cur, next);
        qint8_t *tmp = cur; cur = next; next = tmp;
    }

    const qint8_t *cls_tok = cur;

    matmul_int8(cls_tok, HEAD_W, HEAD_B, logits_out, 1, OUT_DIM, DMODEL, SC_HEAD_q15);
}
