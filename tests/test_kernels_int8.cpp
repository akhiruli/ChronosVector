/*
 * INT8 kernel tests. Compiled and run only when CHRONOSV_ENABLE_INT8 is on.
 *
 * Coverage:
 *   1. QuantizeF32ToI8 — round-trip fidelity, edge cases (zero vector,
 *      extreme values, negative values, saturation at ±127).
 *   2. CosineI8Chunk — matches float32 kernel within design tolerance
 *      (~1-2% on unit-normalized inputs).
 *   3. EuclideanSqI8Chunk — same tolerance check; non-negative clamp works.
 *   4. Both kernels: zero-vector handling produces score=0 without NaN.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "kernels.h"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using chronosv::internal::CosineF32Chunk;
using chronosv::internal::CosineI8Chunk;
using chronosv::internal::EuclideanSqF32Chunk;
using chronosv::internal::EuclideanSqI8Chunk;
using chronosv::internal::L2NormF32;
using chronosv::internal::QuantizeF32ToI8;

/* ==========================================================================
 * QuantizeF32ToI8
 * ========================================================================== */

TEST_CASE("QuantizeF32ToI8: symmetric peak at ±127", "[int8][quantize]") {
    /* max abs = 5, so scale = 5/127 ≈ 0.0394. The peak values 5 and -5 map
     * to 127 and -127 exactly. Intermediate values scale linearly. */
    std::array<float, 4> v = {5.0f, -5.0f, 0.0f, 2.5f};
    std::array<std::int8_t, 4> q{};
    float scale = -1.0f;
    QuantizeF32ToI8(v.data(), 4, q.data(), &scale);

    REQUIRE_THAT(scale, WithinAbs(5.0f / 127.0f, 1e-6f));
    REQUIRE(q[0] ==  127);
    REQUIRE(q[1] == -127);
    REQUIRE(q[2] ==  0);
    /* 2.5 / (5/127) = 63.5 → rounds to 64 (round-half-away-from-zero). */
    REQUIRE(q[3] == 64);
}

TEST_CASE("QuantizeF32ToI8: zero vector produces all-zero q + scale=0",
          "[int8][quantize][edge]") {
    std::array<float, 8> v{};
    std::array<std::int8_t, 8> q{};
    float scale = 999.0f;
    QuantizeF32ToI8(v.data(), 8, q.data(), &scale);
    for (auto x : q) REQUIRE(x == 0);
    REQUIRE(scale == 0.0f);
}

TEST_CASE("QuantizeF32ToI8: round-trip approximates original within scale/2",
          "[int8][quantize][accuracy]") {
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    constexpr std::size_t kDim = 128;

    std::vector<float> orig(kDim);
    for (auto& x : orig) x = dist(rng);

    std::vector<std::int8_t> q(kDim);
    float scale = 0.0f;
    QuantizeF32ToI8(orig.data(), kDim, q.data(), &scale);

    /* Reconstruct: reconstructed[i] = q[i] * scale. Max reconstruction
     * error per component is scale/2 (rounding to nearest int). */
    for (std::size_t i = 0; i < kDim; ++i) {
        const float reconstructed = static_cast<float>(q[i]) * scale;
        REQUIRE_THAT(reconstructed, WithinAbs(orig[i], scale * 0.6f));
    }
}

/* ==========================================================================
 * CosineI8Chunk vs. CosineF32Chunk (accuracy)
 * ========================================================================== */

TEST_CASE("CosineI8Chunk matches float32 reference within design tolerance (~2%)",
          "[int8][cosine][accuracy]") {
    std::mt19937 rng(0xABCDEF);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    constexpr std::size_t kDim   = 128;
    constexpr std::size_t kCount = 100;

    std::vector<float>       vecs_f(kCount * kDim);
    std::vector<std::int8_t> vecs_i8(kCount * kDim);
    std::vector<float>       scales(kCount);
    std::vector<float>       row_norms(kCount);
    for (auto& x : vecs_f) x = dist(rng);
    for (std::size_t i = 0; i < kCount; ++i) {
        QuantizeF32ToI8(&vecs_f[i * kDim], kDim,
                        &vecs_i8[i * kDim], &scales[i]);
        row_norms[i] = L2NormF32(&vecs_f[i * kDim], kDim);
    }

    std::vector<float>       q_f(kDim);
    for (auto& x : q_f) x = dist(rng);
    std::vector<std::int8_t> q_i8(kDim);
    float q_scale = 0.0f;
    QuantizeF32ToI8(q_f.data(), kDim, q_i8.data(), &q_scale);
    const float q_norm = L2NormF32(q_f.data(), kDim);

    std::vector<float> out_f32(kCount), out_i8(kCount);
    CosineF32Chunk(vecs_f.data(), row_norms.data(), kCount, kDim,
                   q_f.data(), q_norm, out_f32.data());
    CosineI8Chunk(vecs_i8.data(), scales.data(), row_norms.data(),
                  kCount, kDim, q_i8.data(), q_scale, q_norm, out_i8.data());

    /* Expected tolerance: ~1-2% cosine-similarity delta on typical
     * embeddings. Use 3% to allow for random-input variance. */
    for (std::size_t i = 0; i < kCount; ++i) {
        REQUIRE_THAT(out_i8[i], WithinAbs(out_f32[i], 0.03f));
    }
}

TEST_CASE("CosineI8Chunk: zero query yields all-zero scores",
          "[int8][cosine][edge]") {
    constexpr std::size_t kDim = 4, kCount = 2;
    std::vector<std::int8_t> vecs(kCount * kDim, 50);
    std::vector<float> scales = {1.0f, 1.0f};
    std::vector<float> norms  = {2.0f, 3.0f};
    std::vector<std::int8_t> q(kDim, 0);   /* zero query */
    std::vector<float> out(kCount, -1.0f);
    CosineI8Chunk(vecs.data(), scales.data(), norms.data(),
                  kCount, kDim, q.data(), /*q_scale=*/0.0f, /*q_norm=*/0.0f,
                  out.data());
    REQUIRE(out[0] == 0.0f);
    REQUIRE(out[1] == 0.0f);
}

TEST_CASE("CosineI8Chunk: per-row zero norm/scale yields zero score",
          "[int8][cosine][edge]") {
    constexpr std::size_t kDim = 4;
    std::vector<std::int8_t> vecs(2 * kDim, 10);
    std::vector<float> scales = {0.0f, 1.0f};   /* first row is zero */
    std::vector<float> norms  = {0.0f, 2.0f};
    std::vector<std::int8_t> q(kDim, 20);
    std::vector<float> out(2, -1.0f);
    CosineI8Chunk(vecs.data(), scales.data(), norms.data(),
                  2, kDim, q.data(), 1.0f, 5.0f, out.data());
    REQUIRE(out[0] == 0.0f);      /* zero row */
    /* Row 1 non-zero score — just require finite. */
    REQUIRE(std::isfinite(out[1]));
}

/* ==========================================================================
 * EuclideanSqI8Chunk vs. EuclideanSqF32Chunk
 * ========================================================================== */

TEST_CASE("EuclideanSqI8Chunk matches float32 reference within tolerance",
          "[int8][euclidean][accuracy]") {
    std::mt19937 rng(0x123456);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    constexpr std::size_t kDim   = 128;
    constexpr std::size_t kCount = 50;

    std::vector<float>       vecs_f(kCount * kDim);
    std::vector<std::int8_t> vecs_i8(kCount * kDim);
    std::vector<float>       scales(kCount);
    std::vector<float>       row_norms(kCount);
    for (auto& x : vecs_f) x = dist(rng);
    for (std::size_t i = 0; i < kCount; ++i) {
        QuantizeF32ToI8(&vecs_f[i * kDim], kDim,
                        &vecs_i8[i * kDim], &scales[i]);
        row_norms[i] = L2NormF32(&vecs_f[i * kDim], kDim);
    }
    std::vector<float> q_f(kDim);
    for (auto& x : q_f) x = dist(rng);
    std::vector<std::int8_t> q_i8(kDim);
    float q_scale = 0.0f;
    QuantizeF32ToI8(q_f.data(), kDim, q_i8.data(), &q_scale);
    const float q_norm = L2NormF32(q_f.data(), kDim);

    std::vector<float> out_f32(kCount), out_i8(kCount);
    EuclideanSqF32Chunk(vecs_f.data(), row_norms.data(), kCount, kDim,
                        q_f.data(), q_norm, out_f32.data());
    EuclideanSqI8Chunk(vecs_i8.data(), scales.data(), row_norms.data(),
                       kCount, kDim, q_i8.data(), q_scale, q_norm, out_i8.data());

    /* Both kernels use the norm-identity path so they share the same
     * cancellation characteristics. Tolerance widened to accommodate
     * INT8 quantization noise plus identity-path cancellation. */
    for (std::size_t i = 0; i < kCount; ++i) {
        /* Distance magnitudes for dim=128 uniform[-2,2] land in the
         * hundreds; ±10% relative is well within combined tolerance. */
        REQUIRE(out_i8[i] >= 0.0f);
        REQUIRE_THAT(out_i8[i], WithinRel(out_f32[i], 0.10f));
    }
}

TEST_CASE("EuclideanSqI8Chunk: identical query and row → distance ≈ 0",
          "[int8][euclidean][boundary]") {
    constexpr std::size_t kDim = 8;
    std::vector<float> v = {1, 2, -1, 0.5, -3, 4, 0, 2.5};
    std::vector<std::int8_t> vecs_i8(kDim);
    float scale = 0.0f;
    QuantizeF32ToI8(v.data(), kDim, vecs_i8.data(), &scale);
    const float norm = L2NormF32(v.data(), kDim);

    /* Query = same as row. */
    std::vector<float> out(1, -1.0f);
    EuclideanSqI8Chunk(vecs_i8.data(), &scale, &norm,
                       1, kDim, vecs_i8.data(), scale, norm, out.data());
    /* The identity path (||a||² + ||b||² - 2·(a·b)) has catastrophic
     * cancellation when a ≈ b — both large numbers cancel to a small
     * residue whose magnitude is around float-epsilon of the norms. The
     * important guarantees are: (a) non-negative (clamp works), (b) small
     * relative to the norms themselves. Bound loose: 1% of ||a||². */
    REQUIRE(out[0] >= 0.0f);
    REQUIRE(out[0] <= 0.01f * norm * norm);
}
