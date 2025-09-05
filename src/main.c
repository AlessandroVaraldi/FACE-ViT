// main.c — Test TinyViTDet INT8 on a .ppm (P6) image
// Build:  gcc -O3 -std=c11 -Wall -Wextra -o test_vit main.c engine.c kernel.c weights.o -lm
// Run:    ./test_vit input.ppm [--pe-scale 2.0] [--no-resize]
//
// Output:
//   - out_head.bin : raw INT8 tensor [TOKENS, OUT_DIM]
//   - console: top-10 patch tokens by channel-0 score (p_logit raw int8)
//
// Requirements:
//   - model.h (autogenerato dal tuo onnx2int8.py) + weights.o (o simile)
//   - engine.c implementa: void vit_init(void);
//                          void vit_forward(const int8_t*, int8_t*);

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>
#include <math.h>

#include "model.h"   // MODEL_* macros, OFF_*, SC_*, PE_IN_SCALE_Q24 (opz. PE_OUT_SCALE_Q31)
#include "kernel.h"  // sat8 (se vuoi riusarli) - opzionale

#ifndef PE_IN_SCALE_Q24
#define PE_IN_SCALE_Q24 ((int32_t)(2<<24))  // fallback 2.0
#endif

// Prototipi dall'engine
void vit_init(void);
void vit_forward(const int8_t *inp_patches, int8_t *out_head);

// ---------------- CLI helpers ----------------
static int   g_no_resize = 0;
static float g_pe_scale  = 0.0f;   // 0 => usa macro dal header se disponibile

static void parse_args(int argc, char **argv, const char **ppm_path){
    if (argc < 2) {
        fprintf(stderr, "Usage: %s input.ppm [--pe-scale S] [--no-resize]\n", argv[0]);
        exit(1);
    }
    *ppm_path = argv[1];
    for (int i=2; i<argc; ++i){
        if (!strcmp(argv[i], "--no-resize")) { g_no_resize = 1; continue; }
        if (!strcmp(argv[i], "--pe-scale") && i+1 < argc){
            g_pe_scale = (float)atof(argv[++i]);
            continue;
        }
        fprintf(stderr, "Unknown arg: %s\n", argv[i]);
        exit(1);
    }
}

// --------------- PPM loader (P6, 8-bit) ---------------
static int skip_ws_and_comments(FILE *f){
    int c;
    // Salta whitespace e commenti '#'
    do {
        c = fgetc(f);
        if (c == '#'){
            // salta linea
            do { c = fgetc(f); } while (c != '\n' && c != EOF);
        }
    } while (c==' ' || c=='\n' || c=='\r' || c=='\t');
    if (c != EOF) ungetc(c, f);
    return 0;
}
static int read_int(FILE *f){
    skip_ws_and_comments(f);
    int v = 0, c = fgetc(f);
    if (c == EOF) return -1;
    int neg = 0;
    if (c=='-' ){ neg=1; c=fgetc(f); }
    while (c>='0' && c<='9'){
        v = v*10 + (c - '0');
        c = fgetc(f);
    }
    if (c!=EOF) ungetc(c,f);
    return neg ? -v : v;
}

typedef struct { int w,h; uint8_t *rgb; } ImageU8x3;

static ImageU8x3 load_ppm(const char *path){
    ImageU8x3 im = {0,0,NULL};
    FILE *f = fopen(path, "rb");
    if(!f){ perror("fopen"); exit(1); }
    // Header "P6"
    char magic[3]={0};
    if (fread(magic,1,2,f)!=2 || magic[0]!='P' || magic[1]!='6'){
        fprintf(stderr, "PPM must be P6 (binary RGB)\n");
        exit(1);
    }
    int w = read_int(f);
    int h = read_int(f);
    int maxv = read_int(f);
    if (w<=0 || h<=0 || maxv!=255){
        fprintf(stderr, "Invalid PPM header (w=%d h=%d max=%d). Need maxval=255.\n", w,h,maxv);
        exit(1);
    }
    // Un singolo byte di whitespace prima dei dati
    fgetc(f);
    size_t bytes = (size_t)w*(size_t)h*3;
    uint8_t *buf = (uint8_t*)malloc(bytes);
    if(!buf){ fprintf(stderr,"OOM\n"); exit(1); }
    if (fread(buf,1,bytes,f)!=bytes){
        fprintf(stderr,"PPM truncated\n");
        exit(1);
    }
    fclose(f);
    im.w=w; im.h=h; im.rgb=buf;
    return im;
}

// --------------- Resize NN (RGB) ---------------
static inline int nn_index(int x, int src, int dst){
    // nearest: floor((x + 0.5) * src / dst)  -->  ((2x+1)*src) / (2*dst)
    int64_t num = (int64_t)(2*x + 1) * (int64_t)src;
    int idx = (int)( num / (2*dst) );
    if (idx >= src) idx = src - 1;
    return idx;
}

static uint8_t* resize_nn_rgb(const uint8_t *src, int sw, int sh, int dw, int dh){
    uint8_t *dst = (uint8_t*)malloc((size_t)dw*dh*3);
    if(!dst){ fprintf(stderr,"OOM\n"); exit(1); }
    for (int y=0; y<dh; ++y){
        int sy = nn_index(y, sh, dh);
        for (int x=0; x<dw; ++x){
            int sx = nn_index(x, sw, dw);
            const uint8_t *sp = src + ((size_t)sy*sw + sx)*3;
            uint8_t       *dp = dst + ((size_t)y*dw + x)*3;
            dp[0]=sp[0]; dp[1]=sp[1]; dp[2]=sp[2];
        }
    }
    return dst;
}

// --------------- Im2Col: [PATCHES, PATCH_DIM] ---------------
static inline int8_t quant_pe_u8(uint8_t v, float pe_in_scale){
    // q = round( (v - 128) / pe_in_scale ), clamp [-127,127]
    float q = floorf( ((float)((int)v - 128) / pe_in_scale) + 0.5f );
    if (q < -127.f) q = -127.f; else if (q > 127.f) q = 127.f;
    return (int8_t)q;
}

static void im2col_patches_int8(const uint8_t *rgb, int W, int H,
                                int8_t *out_patches,
                                float pe_scale)
{
    const int GRID = MODEL_GRID;
    // PATCH side: prova MODEL_PATCH se presente; altrimenti IMG/GRID
#ifdef MODEL_PATCH
    const int PATCH = MODEL_PATCH;
#else
    const int PATCH = (W / GRID);
#endif
    const int PDIM  = MODEL_PATCH_DIM; // = 3*PATCH*PATCH (tipico)
    (void)H; // assumiamo quadrata W=H=MODEL_IMG

    for (int gy=0; gy<GRID; ++gy){
        for (int gx=0; gx<GRID; ++gx){
            int t = gy*GRID + gx;                  // token index (1..PATCHES) ma qui estraiamo solo patch
            int ox = gx*PATCH;
            int oy = gy*PATCH;
            int8_t *dst = out_patches + (size_t)t * PDIM;

            int idx = 0;
            for (int c=0; c<3; ++c){              // C, kh, kw
                for (int ky=0; ky<PATCH; ++ky){
                    const uint8_t *row = rgb + ((size_t)(oy+ky)*W + ox)*3;
                    for (int kx=0; kx<PATCH; ++kx){
                        uint8_t px = row[(size_t)kx*3 + c];
                        dst[idx++]  = quant_pe_u8(px, pe_scale);
                    }
                }
            }
        }
    }
}

// --------------- Utility: dump bin ---------------
static void dump_bin(const char *path, const void *buf, size_t bytes){
    FILE *f = fopen(path, "wb");
    if(!f){ perror("fopen(dump)"); return; }
    fwrite(buf,1,bytes,f);
    fclose(f);
}

// --------------- Top-K by channel 0 ---------------
typedef struct { int token; int8_t score; } Pair;
static int cmp_pair_desc(const void *a, const void *b){
    int sa = ((const Pair*)a)->score;
    int sb = ((const Pair*)b)->score;
    return (sb - sa);
}

static void smoke_test_matmul(void){
    /* A: 2x4, W: 4x3 (KxN), B: 3 */
    int8_t  A[2*4] = { 1,2,3,4,  -1,-2,-3,-4 };
    int8_t  W[4*3] = { 1,0,1,  0,1,1,  1,1,0,  -1,0,1 }; /* KxN */
    int32_t B[3]   = { 0, 0, 0 };
    int8_t  O[2*3] = { 0 };

    /* Q0.15 ~ 1.0 => 32767 (NON 32768 che overflowa in int16_t) */
    const int16_t sc = 32767;

    /* Firma da kernel.h: M,N,K sono size_t */
    matmul_int8(A, W, B, O, (size_t)2, (size_t)3, (size_t)4, sc);

    fprintf(stderr, "[smoke] matmul O: [%d %d %d | %d %d %d]\n",
            O[0],O[1],O[2], O[3],O[4],O[5]);
}

int main(int argc, char **argv){
    smoke_test_matmul();
    const char *ppm_path = NULL;
    parse_args(argc, argv, &ppm_path);

    // 1) Carica PPM
    ImageU8x3 im = load_ppm(ppm_path);

#ifdef MODEL_IMG
    const int IMG_SIDE = MODEL_IMG;
#else
    // fallback: usa lato maggiore come IMG
    const int IMG_SIDE = (im.w == im.h) ? im.w : (im.w > im.h ? im.w : im.h);
#endif

    // 2) Ridimensiona (se necessario)
    uint8_t *img_res = NULL;
    if (!g_no_resize && (im.w != IMG_SIDE || im.h != IMG_SIDE)){
        img_res = resize_nn_rgb(im.rgb, im.w, im.h, IMG_SIDE, IMG_SIDE);
        free(im.rgb);
        im.rgb = img_res; im.w = im.h = IMG_SIDE;
        fprintf(stderr, "[info] resized to %dx%d\n", im.w, im.h);
    } else if (im.w != IMG_SIDE || im.h != IMG_SIDE){
        fprintf(stderr, "[warn] input is %dx%d but model expects %dx%d; proceeding anyway (may be wrong)\n",
                im.w, im.h, IMG_SIDE, IMG_SIDE);
    }

    // 3) Scale PatchEmbed
    //    - IN scale: quantizzazione input patch (CLI > header > fallback 2.0)
    float pe_in_scale = g_pe_scale;
    if (pe_in_scale <= 0.f){
        pe_in_scale = ((float)PE_IN_SCALE_Q24) / (float)(1u<<24);
    }
    fprintf(stderr, "[info] PatchEmbed IN  scale = %.7f\n", pe_in_scale);
#ifdef PE_OUT_SCALE_Q31
    fprintf(stderr, "[info] PatchEmbed OUT scale = %.9f\n", ((float)PE_OUT_SCALE_Q31)/(float)(1u<<31));
#endif

    // 4) Alloca buffers
    const int PATCHES = MODEL_PATCHES;     // = GRID^2
    const int PDIM    = MODEL_PATCH_DIM;
    const int TOKENS  = MODEL_TOKENS;      // = 1 + PATCHES
    const int OUTDIM  = MODEL_OUT_DIM;

    int8_t *inp_patches = (int8_t*)malloc((size_t)PATCHES * PDIM);
    int8_t *out_head    = (int8_t*)malloc((size_t)TOKENS  * OUTDIM);
    if (!inp_patches || !out_head){ fprintf(stderr,"OOM\n"); return 1; }

    // 5) Im2Col + quant
    im2col_patches_int8(im.rgb, im.w, im.h, inp_patches, pe_in_scale);

    // 6) Inizializza ed esegui il ViT
    vit_init();
    vit_forward(inp_patches, out_head);

    // 7) Salva output grezzo
    dump_bin("out_head.bin", out_head, (size_t)TOKENS * OUTDIM);

    // 8) Stampa top-10 per canale 0 (escludendo CLS t=0)
    Pair *arr = (Pair*)malloc(sizeof(Pair) * (size_t)PATCHES);
    for (int t=1; t<TOKENS; ++t){
        arr[t-1].token = t;
        arr[t-1].score = out_head[(size_t)t*OUTDIM + 0]; // canale 0: p_logit (raw int8)
    }
    qsort(arr, PATCHES, sizeof(Pair), cmp_pair_desc);

    printf("Top-10 tokens by ch0 (raw int8):\n");
    const int GRID = MODEL_GRID;
    for (int i=0; i<10 && i<PATCHES; ++i){
        int t  = arr[i].token;
        int gy = (t-1) / GRID;
        int gx = (t-1) % GRID;
        int8_t s = arr[i].score;
        printf("  #%d  token=%d (gx=%d, gy=%d)  score=%d\n", i+1, t, gx, gy, (int)s);
    }

    // 9) Cleanup
    free(arr);
    free(out_head);
    free(inp_patches);
    free(im.rgb);
    return 0;
}

/*
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS_DEBUG="-fsanitize=address -g"
cmake --build build -j
convert images/shark.jpg -resize 32x32\! -strip -type TrueColor ppm:images/shark32.ppm
./build/alem images/shark32.ppm
*/