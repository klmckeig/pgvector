// filepath: /data/pgvector_klmckeig/src/vector.c

#include "postgres.h"
#include <math.h>
#include "fmgr.h"
#include "vector.h"
#include "vectorutils.h"

PG_MODULE_MAGIC;

PGDLLEXPORT void _PG_init(void);
void _PG_init(void) {
    VectorInit();
}

/* Utility: Check dimensions match */
static inline void CheckDims(Vector *a, Vector *b) {
    if (a->dim != b->dim)
        ereport(ERROR,
            (errcode(ERRCODE_DATA_EXCEPTION),
             errmsg("vector dimension mismatch")));
}

/* Utility: Allocate a new vector */
Vector *CreateVector(int dim) {
    Vector *v = (Vector *) palloc0(VECTOR_SIZE(dim));
    SET_VARSIZE(v, VECTOR_SIZE(dim));
    v->dim = dim;
    return v;
}

/* Addition */
PG_FUNCTION_INFO_V1(vector_add);
Datum vector_add(PG_FUNCTION_ARGS) {
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    CheckDims(a, b);
    Vector *result = CreateVector(a->dim);
    VectorAdd(a->dim, a->x, b->x, result->x);
    PG_RETURN_POINTER(result);
}

/* Subtraction */
PG_FUNCTION_INFO_V1(vector_sub);
Datum vector_sub(PG_FUNCTION_ARGS) {
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    CheckDims(a, b);
    Vector *result = CreateVector(a->dim);
    VectorSubtract(a->dim, a->x, b->x, result->x);
    PG_RETURN_POINTER(result);
}

/* Multiplication */
PG_FUNCTION_INFO_V1(vector_mul);
Datum vector_mul(PG_FUNCTION_ARGS) {
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    CheckDims(a, b);
    Vector *result = CreateVector(a->dim);
    VectorMultiply(a->dim, a->x, b->x, result->x);
    PG_RETURN_POINTER(result);
}

/* L2 normalization */
PG_FUNCTION_INFO_V1(l2_normalize);
Datum l2_normalize(PG_FUNCTION_ARGS) {
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *result = CreateVector(a->dim);
    VectorL2Normalize(a->dim, a->x, result->x);
    PG_RETURN_POINTER(result);
}

/* L2 norm */
PG_FUNCTION_INFO_V1(vector_norm);
Datum vector_norm(PG_FUNCTION_ARGS) {
    Vector *a = PG_GETARG_VECTOR_P(0);
    float norm = sqrt(VectorL2SquaredDistance(a->dim, a->x, a->x));
    PG_RETURN_FLOAT8(norm);
}

/* L2 squared distance */
PG_FUNCTION_INFO_V1(vector_l2_squared_distance);
Datum vector_l2_squared_distance(PG_FUNCTION_ARGS) {
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    CheckDims(a, b);
    float dist = VectorL2SquaredDistance(a->dim, a->x, b->x);
    PG_RETURN_FLOAT8(dist);
}

/* Inner product */
PG_FUNCTION_INFO_V1(inner_product);
Datum inner_product(PG_FUNCTION_ARGS) {
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    CheckDims(a, b);
    float prod = VectorInnerProduct(a->dim, a->x, b->x);
    PG_RETURN_FLOAT8(prod);
}

/* Cosine similarity */
PG_FUNCTION_INFO_V1(cosine_distance);
Datum cosine_distance(PG_FUNCTION_ARGS) {
    Vector *a = PG_GETARG_VECTOR_P(0);
    Vector *b = PG_GETARG_VECTOR_P(1);
    CheckDims(a, b);
    double sim = VectorCosineSimilarity(a->dim, a->x, b->x);
    if (sim > 1.0) sim = 1.0;
    if (sim < -1.0) sim = -1.0;
    PG_RETURN_FLOAT8(1.0 - sim);
}

/* Binary quantization */
PG_FUNCTION_INFO_V1(binary_quantize);
Datum binary_quantize(PG_FUNCTION_ARGS) {
    Vector *a = PG_GETARG_VECTOR_P(0);
    VarBit *result = InitBitVector(a->dim);
    BinaryQuantize(a->dim, a->x, VARBITS(result));
    PG_RETURN_VARBIT_P(result);
}

/* Get vector dimensions */
PG_FUNCTION_INFO_V1(vector_dims);
Datum vector_dims(PG_FUNCTION_ARGS) {
    Vector *a = PG_GETARG_VECTOR_P(0);
    PG_RETURN_INT32(a->dim);
}