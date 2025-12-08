#ifndef VECTORUTILS_H
#define VECTORUTILS_H

#include "vector.h"

extern void (*BinaryQuantize) (int dim, float *ax, unsigned char *rx);
extern double (*VectorCosineSimilarity)(int dim, float *ax, float *bx);
extern void (*VectorAdd)(int dim, float *ax, float *bx, float *rx);
extern void (*VectorSubtract)(int dim, float *ax, float *bx, float *rx);
extern void (*VectorMultiply)(int dim, float *ax, float *bx, float *rx);
extern void (*VectorL2Normalize)(int dim, float *ax, float *rx);
extern float (*VectorL2SquaredDistance)(int dim, float *ax, float *bx);
extern float (*VectorInnerProduct)(int dim, float *ax, float *bx);

void		VectorInit(void);

#endif
