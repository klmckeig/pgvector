#include "postgres.h"

#include "vectorutils.h"
#include "halfvec.h"			/* for USE_DISPATCH and USE_TARGET_CLONES */

#if defined(USE_DISPATCH)
#define VECTOR_DISPATCH
#endif

#ifdef VECTOR_DISPATCH
#include <immintrin.h>

#if defined(USE__GET_CPUID)
#include <cpuid.h>
#else
#include <intrin.h>
#endif

#ifdef _MSC_VER
#define TARGET_AVX512
#define TARGET_AVX512_GFNI
#else
#define TARGET_AVX512 __attribute__((target("avx512f")))
#define TARGET_AVX512_GFNI __attribute__((target("avx512f,avx512vl,gfni")))
#endif
#endif

/* ========== FUNCTION POINTERS ========== */

// Binary quantization
void		(*BinaryQuantize) (int dim, float *ax, unsigned char *rx);

// Vector math operations
double		(*VectorCosineSimilarity) (int dim, float *ax, float *bx);
void		(*VectorAdd) (int dim, float *ax, float *bx, float *rx);
void		(*VectorSubtract) (int dim, float *ax, float *bx, float *rx);
void		(*VectorMultiply) (int dim, float *ax, float *bx, float *rx);
void		(*VectorL2Normalize) (int dim, float *ax, float *rx);
float		(*VectorL2SquaredDistance) (int dim, float *ax, float *bx);
float		(*VectorInnerProduct) (int dim, float *ax, float *bx);

/* ========== DEFAULT IMPLEMENTATIONS ========== */

static void
BinaryQuantizeDefault(int dim, float *ax, unsigned char *rx) {
    int			i;
    int			count = (dim / 8) * 8;
    unsigned char result_byte;

    for (i = 0; i < count; i += 8)
    {
        result_byte = 0;
        for (int j = 0; j < 8; j++)
            result_byte |= (ax[i + j] > 0) << (7 - j);
        rx[i / 8] = result_byte;
    }
    for (; i < dim; i++)
        rx[i / 8] |= (ax[i] > 0) << (7 - (i % 8));
}

static double
VectorCosineSimilarityDefault(int dim, float *ax, float *bx)
{
    float similarity = 0.0;
    float norma = 0.0;
    float normb = 0.0;

    for (int i = 0; i < dim; i++)
    {
        similarity += ax[i] * bx[i];
        norma += ax[i] * ax[i];
        normb += bx[i] * bx[i];
    }

    return (double) similarity / sqrt((double) norma * (double) normb);
}

static void
VectorAddDefault(int dim, float *ax, float *bx, float *rx)
{
    for (int i = 0; i < dim; i++)
    {
        rx[i] = ax[i] + bx[i];
        if (isinf(rx[i]))
            ereport(ERROR, (errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
                            errmsg("vector addition overflow")));
    }
}

static void
VectorSubtractDefault(int dim, float *ax, float *bx, float *rx)
{
    for (int i = 0; i < dim; i++)
    {
        rx[i] = ax[i] - bx[i];
        if (isinf(rx[i]))
            ereport(ERROR, (errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
                            errmsg("vector subtraction overflow")));
    }
}

static void
VectorMultiplyDefault(int dim, float *ax, float *bx, float *rx)
{
    for (int i = 0; i < dim; i++)
    {
        rx[i] = ax[i] * bx[i];
        if (isinf(rx[i]))
            ereport(ERROR, (errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
                            errmsg("vector multiplication overflow")));
    }
}

static void
VectorL2NormalizeDefault(int dim, float *ax, float *rx)
{
    double norm = 0;

    for (int i = 0; i < dim; i++)
        norm += (double) ax[i] * (double) ax[i];

    norm = sqrt(norm);

    if (norm > 0)
    {
        for (int i = 0; i < dim; i++)
        {
            rx[i] = ax[i] / norm;
            if (isinf(rx[i]))
                ereport(ERROR, (errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
                                errmsg("normalization overflow")));
        }
    }
    else
    {
        memset(rx, 0, dim * sizeof(float));
    }
}

static float
VectorL2SquaredDistanceDefault(int dim, float *ax, float *bx)
{
    float distance = 0.0;

    for (int i = 0; i < dim; i++)
    {
        float diff = ax[i] - bx[i];
        distance += diff * diff;
    }

    return distance;
}

static float
VectorInnerProductDefault(int dim, float *ax, float *bx)
{
    float distance = 0.0;

    for (int i = 0; i < dim; i++)
        distance += ax[i] * bx[i];

    return distance;
}

/* ========== AVX-512 OPTIMIZED IMPLEMENTATIONS ========== */

#ifdef VECTOR_DISPATCH

// Existing binary quantization functions
TARGET_AVX512 static inline void
BinaryQuantizeAvx512Compare(int dim, float *ax, unsigned char *rx) {
    int				rx_bytes = 0;
	unsigned long	mask;
	__m512			axi_512;
	__m512			zero_512 = _mm512_setzero_ps();
	__mmask16		cmp;

	for (int i = 0; i < dim; i += 16)
	{
		if (dim - i < 16)
		{
			mask = (1 << (dim - i)) - 1;
			axi_512 = _mm512_maskz_loadu_ps(mask, ax + i);
			cmp = _mm512_cmp_ps_mask(axi_512, zero_512, _CMP_GT_OQ);
			if (dim - i > 8)
				*((uint16_t*)(rx + rx_bytes)) = cmp;
			else {
				*((uint8_t*)(rx + rx_bytes)) = (uint8_t)(cmp & 0xFF);
			}
		}
		else
		{
			axi_512 = _mm512_loadu_ps(ax + i);
			cmp = _mm512_cmp_ps_mask(axi_512, zero_512, _CMP_GT_OQ);
			*((uint16_t*)(rx + rx_bytes)) = cmp;
			rx_bytes += 2;
		}
	}
}

static const uint8_t bit_invert_lookup[16] = {
    0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE,
    0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF
};

TARGET_AVX512 static void
BinaryQuantizeAvx512(int dim, float *ax, unsigned char *rx) {
    int	rx_bytes = 0;

	BinaryQuantizeAvx512Compare(dim, ax, rx);

	rx_bytes = dim / 8;
	if (dim % 8 > 0)
		rx_bytes++;
	for (int i = 0; i < rx_bytes; i++)
		rx[i] = (bit_invert_lookup[rx[i] & 0b1111] << 4) | bit_invert_lookup[rx[i] >> 4];
}

#define GFNI_REVBIT		0x8040201008040201

TARGET_AVX512_GFNI static void
BinaryQuantizeAvx512Gfni(int dim, float *ax, unsigned char *rx) {
    int			rx_bytes = 0;
	__m128i		revbit = _mm_set1_epi64x(GFNI_REVBIT);
	__m128i		rxi;
	__m128i		rxirev;
	int			count;
	int			i = 0;

	BinaryQuantizeAvx512Compare(dim, ax, rx);

	rx_bytes = dim / 8;
	if (dim % 8 > 0)
		rx_bytes++;
	count = (rx_bytes/16)*16;
	for (; i < count; i += 16)
	{
		rxi = _mm_loadu_epi64(rx + i);
		rxirev = _mm_gf2p8affine_epi64_epi8(rxi, revbit, 0);
		_mm_storeu_epi64(rx + i, rxirev);
	}
	for (; i < rx_bytes; i++)
		rx[i] =(bit_invert_lookup[rx[i] & 0b1111] << 4) | bit_invert_lookup[rx[i] >> 4];
}

// New vector math optimizations
TARGET_AVX512 static double
VectorCosineSimilarityAvx512(int dim, float *ax, float *bx)
{
    __m512 sim_acc = _mm512_setzero_ps();
    __m512 norma_acc = _mm512_setzero_ps();
    __m512 normb_acc = _mm512_setzero_ps();
    int count = (dim / 16) * 16;

    // Process 16 floats at a time
    for (int i = 0; i < count; i += 16)
    {
        __m512 a = _mm512_loadu_ps(&ax[i]);
        __m512 b = _mm512_loadu_ps(&bx[i]);

        // Three FMA operations simultaneously
        sim_acc = _mm512_fmadd_ps(a, b, sim_acc);      // ax[i] * bx[i] + sim_acc
        norma_acc = _mm512_fmadd_ps(a, a, norma_acc);  // ax[i] * ax[i] + norma_acc
        normb_acc = _mm512_fmadd_ps(b, b, normb_acc);  // bx[i] * bx[i] + normb_acc
    }

    // Horizontal reduction to scalars
    float similarity = _mm512_reduce_add_ps(sim_acc);
    float norma = _mm512_reduce_add_ps(norma_acc);
    float normb = _mm512_reduce_add_ps(normb_acc);

    // Handle remaining elements (< 16)
    for (int i = count; i < dim; i++)
    {
        similarity += ax[i] * bx[i];
        norma += ax[i] * ax[i];
        normb += bx[i] * bx[i];
    }

    return (double) similarity / sqrt((double) norma * (double) normb);
}

TARGET_AVX512 static void
VectorAddAvx512(int dim, float *ax, float *bx, float *rx)
{
    int count = (dim / 16) * 16;

    // Process 16 floats at a time
    for (int i = 0; i < count; i += 16)
    {
        __m512 a = _mm512_loadu_ps(&ax[i]);
        __m512 b = _mm512_loadu_ps(&bx[i]);
        __m512 result = _mm512_add_ps(a, b);

        // Vectorized overflow check
        __mmask16 inf_mask = _mm512_fpclass_ps_mask(result, 0x88); // INF class
        if (inf_mask)
            ereport(ERROR, (errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
                            errmsg("vector addition overflow")));

        _mm512_storeu_ps(&rx[i], result);
    }

    // Handle remaining elements
    for (int i = count; i < dim; i++)
    {
        rx[i] = ax[i] + bx[i];
        if (isinf(rx[i]))
            ereport(ERROR, (errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
                            errmsg("vector addition overflow")));
    }
}

TARGET_AVX512 static void
VectorSubtractAvx512(int dim, float *ax, float *bx, float *rx)
{
    int count = (dim / 16) * 16;

    for (int i = 0; i < count; i += 16)
    {
        __m512 a = _mm512_loadu_ps(&ax[i]);
        __m512 b = _mm512_loadu_ps(&bx[i]);
        __m512 result = _mm512_sub_ps(a, b);

        __mmask16 inf_mask = _mm512_fpclass_ps_mask(result, 0x88);
        if (inf_mask)
            ereport(ERROR, (errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
                            errmsg("vector subtraction overflow")));

        _mm512_storeu_ps(&rx[i], result);
    }

    for (int i = count; i < dim; i++)
    {
        rx[i] = ax[i] - bx[i];
        if (isinf(rx[i]))
            ereport(ERROR, (errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
                            errmsg("vector subtraction overflow")));
    }
}

TARGET_AVX512 static void
VectorMultiplyAvx512(int dim, float *ax, float *bx, float *rx)
{
    int count = (dim / 16) * 16;

    for (int i = 0; i < count; i += 16)
    {
        __m512 a = _mm512_loadu_ps(&ax[i]);
        __m512 b = _mm512_loadu_ps(&bx[i]);
        __m512 result = _mm512_mul_ps(a, b);

        __mmask16 inf_mask = _mm512_fpclass_ps_mask(result, 0x88);
        if (inf_mask)
            ereport(ERROR, (errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
                            errmsg("vector multiplication overflow")));

        _mm512_storeu_ps(&rx[i], result);
    }

    for (int i = count; i < dim; i++)
    {
        rx[i] = ax[i] * bx[i];
        if (isinf(rx[i]))
            ereport(ERROR, (errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
                            errmsg("vector multiplication overflow")));
    }
}

TARGET_AVX512 static void
VectorL2NormalizeAvx512(int dim, float *ax, float *rx)
{
    __m512 norm_acc = _mm512_setzero_ps();
    int count = (dim / 16) * 16;

    // First pass: compute norm
    for (int i = 0; i < count; i += 16)
    {
        __m512 a = _mm512_loadu_ps(&ax[i]);
        norm_acc = _mm512_fmadd_ps(a, a, norm_acc);
    }

    float norm_squared = _mm512_reduce_add_ps(norm_acc);

    // Handle remaining elements for norm
    for (int i = count; i < dim; i++)
        norm_squared += ax[i] * ax[i];

    float norm = sqrt(norm_squared);

    if (norm > 0)
    {
        __m512 inv_norm = _mm512_set1_ps(1.0f / norm);

        // Second pass: normalize
        for (int i = 0; i < count; i += 16)
        {
            __m512 a = _mm512_loadu_ps(&ax[i]);
            __m512 result = _mm512_mul_ps(a, inv_norm);

            __mmask16 inf_mask = _mm512_fpclass_ps_mask(result, 0x88);
            if (inf_mask)
                ereport(ERROR, (errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
                                errmsg("normalization overflow")));

            _mm512_storeu_ps(&rx[i], result);
        }

        // Handle remaining elements
        for (int i = count; i < dim; i++)
        {
            rx[i] = ax[i] / norm;
            if (isinf(rx[i]))
                ereport(ERROR, (errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
                                errmsg("normalization overflow")));
        }
    }
    else
    {
        memset(rx, 0, dim * sizeof(float));
    }
}

TARGET_AVX512 static float
VectorL2SquaredDistanceAvx512(int dim, float *ax, float *bx)
{
    __m512 dist_acc = _mm512_setzero_ps();
    int count = (dim / 16) * 16;

    for (int i = 0; i < count; i += 16)
    {
        __m512 a = _mm512_loadu_ps(&ax[i]);
        __m512 b = _mm512_loadu_ps(&bx[i]);
        __m512 diff = _mm512_sub_ps(a, b);
        dist_acc = _mm512_fmadd_ps(diff, diff, dist_acc);
    }

    float distance = _mm512_reduce_add_ps(dist_acc);

    // Handle remaining elements
    for (int i = count; i < dim; i++)
    {
        float diff = ax[i] - bx[i];
        distance += diff * diff;
    }

    return distance;
}

TARGET_AVX512 static float
VectorInnerProductAvx512(int dim, float *ax, float *bx)
{
    __m512 prod_acc = _mm512_setzero_ps();
    int count = (dim / 16) * 16;

    for (int i = 0; i < count; i += 16)
    {
        __m512 a = _mm512_loadu_ps(&ax[i]);
        __m512 b = _mm512_loadu_ps(&bx[i]);
        prod_acc = _mm512_fmadd_ps(a, b, prod_acc);
    }

    float distance = _mm512_reduce_add_ps(prod_acc);

    // Handle remaining elements
    for (int i = count; i < dim; i++)
        distance += ax[i] * bx[i];

    return distance;
}

/* ========== CPU FEATURE DETECTION ========== */

#define CPU_FEATURE_OSXSAVE  (1 << 27)
#define CPU_FEATURE_AVX512F  (1 << 16)
#define CPU_FEATURE_AVX512VL (1 << 31)
#define CPU_FEATURE_GFNI     (1 << 8)

#ifdef _MSC_VER
#define TARGET_XSAVE
#else
#define TARGET_XSAVE __attribute__((target("xsave")))
#endif

TARGET_XSAVE static bool
SupportsOsXsave()
{
    unsigned int exx[4] = {0, 0, 0, 0};

#if defined(HAVE__GET_CPUID)
    __get_cpuid(1, &exx[0], &exx[1], &exx[2], &exx[3]);
#else
    __cpuid(exx, 1);
#endif

    return (exx[2] & CPU_FEATURE_OSXSAVE) == CPU_FEATURE_OSXSAVE;
}

TARGET_XSAVE static bool
SupportsAvx512(unsigned int feature)
{
    unsigned int exx[4] = {0, 0, 0, 0};

    /* Check OS supports XSAVE */
    if (!SupportsOsXsave())
        return false;

    /* Check XMM, YMM, and ZMM registers are enabled */
    if ((_xgetbv(0) & 0xe6) != 0xe6)
        return false;

#if defined(HAVE__GET_CPUID)
    __get_cpuid_count(7, 0, &exx[0], &exx[1], &exx[2], &exx[3]);
#elif defined(HAVE__CPUID)
    __cpuid(exx, 7, 0);
#endif

    return (exx[1] & feature) == feature;
}

TARGET_XSAVE static bool
SupportsGfni()
{
    unsigned int exx[4] = {0, 0, 0, 0};

#if defined(HAVE__GET_CPUID)
    __get_cpuid_count(7, 0, &exx[0], &exx[1], &exx[2], &exx[3]);
#elif defined(HAVE__CPUID)
    __cpuid(exx, 7, 0);
#endif

    return (exx[2] & CPU_FEATURE_GFNI) == CPU_FEATURE_GFNI;
}

#endif /* VECTOR_DISPATCH */

/* ========== INITIALIZATION ========== */

void
VectorInit(void)
{
    /* Initialize binary quantization */
    BinaryQuantize = BinaryQuantizeDefault;

    /* Initialize vector math operations */
    VectorCosineSimilarity = VectorCosineSimilarityDefault;
    VectorAdd = VectorAddDefault;
    VectorSubtract = VectorSubtractDefault;
    VectorMultiply = VectorMultiplyDefault;
    VectorL2Normalize = VectorL2NormalizeDefault;
    VectorL2SquaredDistance = VectorL2SquaredDistanceDefault;
    VectorInnerProduct = VectorInnerProductDefault;

#ifdef VECTOR_DISPATCH
    /* Detect CPU capabilities and select optimized implementations */
    bool has_avx512f = SupportsAvx512(CPU_FEATURE_AVX512F);
    bool has_avx512vl = SupportsAvx512(CPU_FEATURE_AVX512F | CPU_FEATURE_AVX512VL);
    bool has_gfni = SupportsGfni();

    if (has_avx512f)
    {
        /* Basic AVX-512 optimizations */
        VectorCosineSimilarity = VectorCosineSimilarityAvx512;
        VectorAdd = VectorAddAvx512;
        VectorSubtract = VectorSubtractAvx512;
        VectorMultiply = VectorMultiplyAvx512;
        VectorL2Normalize = VectorL2NormalizeAvx512;
        VectorL2SquaredDistance = VectorL2SquaredDistanceAvx512;
        VectorInnerProduct = VectorInnerProductAvx512;

        /* Binary quantization with basic AVX-512 */
        BinaryQuantize = BinaryQuantizeAvx512;
    }

    if (has_avx512vl && has_gfni)
    {
        /* Enhanced binary quantization with GFNI */
        BinaryQuantize = BinaryQuantizeAvx512Gfni;
    }
#endif
}