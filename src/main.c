// main.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#ifdef XHEEP

#else

  #define FRAME_W  32
  #define FRAME_H  32
  #define FRAME_C   3
  static int load_ppm(const char *path, uint8_t *buf) {
      FILE *f = fopen(path, "rb");
      if (!f) { perror("fopen"); return -1; }
      int w, h, maxv;
      char header[3];
      if (fscanf(f, "%2s\n%d %d\n%d\n", header, &w, &h, &maxv) != 4) {
          fprintf(stderr, "PPM header error\n"); fclose(f); return -1;
      }
      if (strcmp(header,"P6") || w!=FRAME_W || h!=FRAME_H || maxv!=255) {
          fprintf(stderr, "PPM must be P6 %dx%d 255\n", FRAME_W, FRAME_H);
          fclose(f); return -1;
      }
      size_t got = fread(buf, 1, FRAME_W*FRAME_H*FRAME_C, f);
      fclose(f);
      return got == FRAME_W*FRAME_H*FRAME_C ? 0 : -1;
  }
#endif

#include "kernel.h"    // qint8_t
#include "model.h"     // MODEL_PATCHES, MODEL_PATCH_DIM, MODEL_OUT_DIM
#include "engine.h"    // vit_init(), vit_forward()

#define INP_SIZE  (MODEL_PATCHES * MODEL_PATCH_DIM)
#define OUT_SIZE  MODEL_OUT_DIM

static const char *class_names[MODEL_OUT_DIM] = {
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee",
    "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
    "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
    "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant",
    "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo",
    "computer_keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter",
    "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate",
    "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road",
    "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk",
    "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar",
    "sunflower", "sweet_pepper", "table", "tank", "telephone",
    "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm"
};

int main(int argc, char **argv) {
    #ifndef XHEEP
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <image.ppm>\n", argv[0]);
        return 1;
    }
    #endif

    uint8_t rgb[INP_SIZE];

    #ifdef XHEEP

    #else

    if (load_ppm(argv[1], rgb) < 0) {
        fprintf(stderr, "Errore caricamento immagine\n");
        return 1;
    }

    #endif

    vit_init();

    qint8_t in_q[INP_SIZE];
    for (int i = 0; i < INP_SIZE; i++)
        in_q[i] = (qint8_t)(rgb[i] - 128);

    qint8_t out_q[OUT_SIZE];
    vit_forward(in_q, out_q);

    qint8_t prob_q[OUT_SIZE];
    softmax_int8(out_q, prob_q, OUT_SIZE);

    /* -------------------------------------------------------------- */
    /*  Pick the most probable class (largest logit)                  */
    /* -------------------------------------------------------------- */
    int best = 0;
    for (int i = 1; i < OUT_SIZE; i++)
        if (prob_q[i] > prob_q[best])
            best = i;

    printf("%d %s\n", best, class_names[best]);

    /* -------------------------------------------------------------- */
    /* Print all probabilities (0..127)                               */
    /* -------------------------------------------------------------- */
    for (int i = 0; i < OUT_SIZE; i++)
        printf("%d %s: %d\n", i, class_names[i], prob_q[i]);
    printf("\n");

    return 0;

    /*
        rm -rf build
        cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS_DEBUG="-fsanitize=address -g"
        cmake --build build -j
        convert images/shark.jpg -resize 32x32\! -strip -type TrueColor ppm:images/shark32.ppm
        ./build/alem images/shark32.ppm
    */
}
