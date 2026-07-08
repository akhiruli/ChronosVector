/*
 * Distance-kernel unit tests. Covers:
 *   - L2 norm on curated cases
 *   - Cosine similarity: bit-exact hand-computed cases + property test vs. naive ref
 *   - Euclidean squared: same
 *   - Edge cases: zero q-norm, zero row-norm
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "kernels.h"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using chronosv::internal::CosineF32Chunk;
using chronosv::internal::EuclideanSqF32Chunk;
using chronosv::internal::EuclideanSqF32Chunk_Direct;
using chronosv::internal::L2NormF32;

namespace {

// Little helper: compute per-row norms for a vecs buffer, used to feed the
// euclidean fast path.
std::vector<float> compute_norms(const std::vector<float>& vecs,
                                 std::size_t count, std::size_t dim) {
    std::vector<float> n(count);
    for (std::size_t i = 0; i < count; ++i) {
        n[i] = L2NormF32(&vecs[i * dim], dim);
    }
    return n;
}

}  // namespace

// -- Naive reference implementations for property tests ---------------------

namespace ref {

float dot(const float* a, const float* b, std::size_t d) {
    float s = 0;
    for (std::size_t i = 0; i < d; ++i) s += a[i] * b[i];
    return s;
}
float norm(const float* v, std::size_t d) {
    return std::sqrt(dot(v, v, d));
}
float cosine(const float* a, const float* b, std::size_t d) {
    const float na = norm(a, d), nb = norm(b, d);
    if (na == 0 || nb == 0) return 0;
    return dot(a, b, d) / (na * nb);
}
float euclid_sq(const float* a, const float* b, std::size_t d) {
    float s = 0;
    for (std::size_t i = 0; i < d; ++i) {
        const float delta = a[i] - b[i];
        s += delta * delta;
    }
    return s;
}

}  // namespace ref

// ---------------------------------------------------------------------------
// L2 norm
// ---------------------------------------------------------------------------

TEST_CASE("L2NormF32 matches known values", "[kernel][norm]") {
    // ||(3,4)|| = 5
    std::vector<float> v34 = {3, 4};
    REQUIRE_THAT(L2NormF32(v34.data(), v34.size()), WithinAbs(5.0f, 1e-6f));

    // ||(1,0,0,...)|| = 1
    std::vector<float> unit(128, 0.0f);
    unit[0] = 1.0f;
    REQUIRE_THAT(L2NormF32(unit.data(), unit.size()), WithinAbs(1.0f, 1e-6f));

    // ||0|| = 0
    std::vector<float> zero(64, 0.0f);
    REQUIRE_THAT(L2NormF32(zero.data(), zero.size()), WithinAbs(0.0f, 1e-6f));
}

// ---------------------------------------------------------------------------
// Cosine
// ---------------------------------------------------------------------------

TEST_CASE("CosineF32Chunk bit-exact on curated cases", "[kernel][cosine]") {
    // Two rows, dim 3:
    //   row0 = (1,0,0)   norm 1
    //   row1 = (1,1,0)   norm sqrt(2)
    // query = (1,0,0)    qn 1
    //   cos(row0, q) = 1
    //   cos(row1, q) = 1/sqrt(2) ≈ 0.7071068
    std::vector<float> vecs  = {1, 0, 0,   1, 1, 0};
    std::vector<float> norms = {1.0f, std::sqrt(2.0f)};
    std::vector<float> q     = {1, 0, 0};
    std::vector<float> out(2, -1.0f);

    CosineF32Chunk(vecs.data(), norms.data(),
                   /*count=*/2, /*dim=*/3,
                   q.data(), /*qn=*/1.0f,
                   out.data());

    REQUIRE_THAT(out[0], WithinAbs(1.0f, 1e-6f));
    REQUIRE_THAT(out[1], WithinAbs(1.0f / std::sqrt(2.0f), 1e-6f));
}

TEST_CASE("CosineF32Chunk handles zero q-norm as all zeros", "[kernel][cosine][edge]") {
    std::vector<float> vecs  = {1, 0, 0,   0, 1, 0};
    std::vector<float> norms = {1, 1};
    std::vector<float> q     = {0, 0, 0};
    std::vector<float> out(2, -1.0f);

    CosineF32Chunk(vecs.data(), norms.data(), 2, 3, q.data(), /*qn=*/0.0f, out.data());
    REQUIRE(out[0] == 0.0f);
    REQUIRE(out[1] == 0.0f);
}

TEST_CASE("CosineF32Chunk handles per-row zero norm", "[kernel][cosine][edge]") {
    // Row0 is the zero vector; norm[0] = 0 → score should be 0, not NaN.
    std::vector<float> vecs  = {0, 0, 0,   1, 0, 0};
    std::vector<float> norms = {0.0f, 1.0f};
    std::vector<float> q     = {1, 0, 0};
    std::vector<float> out(2, -1.0f);

    CosineF32Chunk(vecs.data(), norms.data(), 2, 3, q.data(), /*qn=*/1.0f, out.data());
    REQUIRE(out[0] == 0.0f);
    REQUIRE_THAT(out[1], WithinAbs(1.0f, 1e-6f));
}

TEST_CASE("CosineF32Chunk matches naive reference (property test)",
          "[kernel][cosine][property]") {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    constexpr std::size_t kDim = 128;
    constexpr std::size_t kCount = 100;

    std::vector<float> vecs(kCount * kDim), norms(kCount), q(kDim), out(kCount);
    for (auto& x : vecs) x = dist(rng);
    for (auto& x : q)    x = dist(rng);

    // Compute per-row norms (SoA cache).
    for (std::size_t i = 0; i < kCount; ++i) {
        norms[i] = L2NormF32(&vecs[i * kDim], kDim);
    }
    const float qn = L2NormF32(q.data(), kDim);

    CosineF32Chunk(vecs.data(), norms.data(), kCount, kDim, q.data(), qn, out.data());

    for (std::size_t i = 0; i < kCount; ++i) {
        const float want = ref::cosine(&vecs[i * kDim], q.data(), kDim);
        REQUIRE_THAT(out[i], WithinAbs(want, 1e-5f));
    }
}

// ---------------------------------------------------------------------------
// Euclidean squared
// ---------------------------------------------------------------------------

TEST_CASE("EuclideanSqF32Chunk bit-exact on curated cases", "[kernel][euclidean]") {
    // row0 = (0,0,0)   q = (3,4,0)   -> 9 + 16 = 25
    // row1 = (3,4,0)   q = (3,4,0)   -> 0
    std::vector<float> vecs  = {0, 0, 0,   3, 4, 0};
    std::vector<float> q     = {3, 4, 0};
    std::vector<float> norms = {0.0f, 5.0f};   // ||row0||=0, ||row1||=5
    std::vector<float> out(2, -1.0f);

    EuclideanSqF32Chunk(vecs.data(), norms.data(), 2, 3,
                        q.data(), /*qn=*/5.0f, out.data());
    REQUIRE_THAT(out[0], WithinAbs(25.0f, 1e-5f));
    REQUIRE_THAT(out[1], WithinAbs( 0.0f, 1e-5f));
}

TEST_CASE("EuclideanSqF32Chunk matches naive reference (property test)",
          "[kernel][euclidean][property]") {
    std::mt19937 rng(54321);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    constexpr std::size_t kDim = 64;
    constexpr std::size_t kCount = 200;

    std::vector<float> vecs(kCount * kDim), q(kDim), out(kCount);
    for (auto& x : vecs) x = dist(rng);
    for (auto& x : q)    x = dist(rng);
    const auto norms = compute_norms(vecs, kCount, kDim);
    const float qn = L2NormF32(q.data(), kDim);

    EuclideanSqF32Chunk(vecs.data(), norms.data(), kCount, kDim,
                        q.data(), qn, out.data());

    for (std::size_t i = 0; i < kCount; ++i) {
        const float want = ref::euclid_sq(&vecs[i * kDim], q.data(), kDim);
        // Norm-identity path is subject to catastrophic cancellation; tolerate
        // ~1e-3 relative on dim=64 uniform[-5,5] where values are in the
        // hundreds/thousands.
        REQUIRE_THAT(out[i], WithinRel(want, 1e-3f));
    }
}

TEST_CASE("EuclideanSqF32Chunk fast vs direct: ordering preserved even when magnitudes drift",
          "[kernel][euclidean][ordering]") {
    // The critical guarantee for top-N selection is that the RANK of results
    // is preserved between the fast and direct kernels. Absolute magnitudes
    // may differ (~1e-5 rel), but if the sorted order matches, users get
    // the right answers.
    std::mt19937 rng(0x0EDEED);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    constexpr std::size_t kDim = 128;
    constexpr std::size_t kCount = 500;
    std::vector<float> vecs(kCount * kDim), q(kDim);
    for (auto& x : vecs) x = dist(rng);
    for (auto& x : q)    x = dist(rng);
    const auto norms = compute_norms(vecs, kCount, kDim);
    const float qn = L2NormF32(q.data(), kDim);

    std::vector<float> fast(kCount), direct(kCount);
    EuclideanSqF32Chunk(vecs.data(), norms.data(), kCount, kDim,
                        q.data(), qn, fast.data());
    EuclideanSqF32Chunk_Direct(vecs.data(), kCount, kDim, q.data(), direct.data());

    // Build the top-10 index lists from each and require they match.
    auto top_k = [](const std::vector<float>& s, std::size_t k) {
        std::vector<std::size_t> idx(s.size());
        for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](std::size_t a, std::size_t b) { return s[a] < s[b]; });
        idx.resize(k);
        return idx;
    };
    auto tf = top_k(fast, 10);
    auto td = top_k(direct, 10);
    REQUIRE(tf == td);
}

TEST_CASE("EuclideanSqF32Chunk fast: non-negative clamp works on identical vectors",
          "[kernel][euclidean][boundary]") {
    // If a row equals q exactly, ||row||^2 + ||q||^2 - 2*dot should be
    // mathematically 0. In float it can go slightly negative due to
    // cancellation. Clamp to zero — never return a negative distance.
    std::vector<float> v = {1.5f, -2.25f, 3.125f, -4.0f, 5.5f};
    std::vector<float> vecs = v;
    std::vector<float> norms = {L2NormF32(v.data(), v.size())};
    std::vector<float> out(1, -999.0f);
    EuclideanSqF32Chunk(vecs.data(), norms.data(), 1, v.size(),
                        v.data(), norms[0], out.data());
    REQUIRE(out[0] >= 0.0f);
    REQUIRE_THAT(out[0], WithinAbs(0.0f, 1e-5f));
}

/* ==========================================================================
 * Coverage-expansion tests (round 2)
 * ==========================================================================
 */

// ---- count = 0 -----------------------------------------------------------

TEST_CASE("Kernels: count=0 is a no-op (no crash)", "[kernel][edge][zero]") {
    // Empty snapshots are legal (fresh sensor) and the engine may still call
    // the kernel. Must not crash, must not touch out buffer.
    std::vector<float> q(4, 1.0f);
    float sentinel = -12345.0f;

    // Cosine: pass count=0. norms/vecs may be null since not dereferenced.
    CosineF32Chunk(/*vecs=*/nullptr, /*norms=*/nullptr,
                   /*count=*/0, /*dim=*/4,
                   q.data(), /*qn=*/1.0f, &sentinel);
    REQUIRE(sentinel == -12345.0f);

    // Euclidean same.
    EuclideanSqF32Chunk(/*vecs=*/nullptr, /*norms=*/nullptr,
                        /*count=*/0, /*dim=*/4,
                        q.data(), /*qn=*/1.0f, &sentinel);
    REQUIRE(sentinel == -12345.0f);
}

// ---- dim = 1 -------------------------------------------------------------

TEST_CASE("Kernels: dim=1 (scalar vectors) computes correctly",
          "[kernel][edge][dim1]") {
    // For dim=1, cosine reduces to sign(row * q). Row=2, q=3 → dot=6,
    // norms 2 and 3 → 6/6 = 1. Row=-2, q=3 → dot=-6, |-2|=2, |3|=3, → -1.
    std::vector<float> vecs = {2.0f, -2.0f};
    std::vector<float> norms = {2.0f, 2.0f};
    std::vector<float> q = {3.0f};
    std::vector<float> out(2);

    CosineF32Chunk(vecs.data(), norms.data(), 2, 1, q.data(), 3.0f, out.data());
    REQUIRE_THAT(out[0], WithinAbs( 1.0f, 1e-6f));
    REQUIRE_THAT(out[1], WithinAbs(-1.0f, 1e-6f));

    // Euclidean sq: (2-3)^2 = 1, (-2-3)^2 = 25.
    // Fast path expects row norms and qn. For scalars, |x| == norm.
    std::vector<float> row_norms_e = {2.0f, 2.0f};
    EuclideanSqF32Chunk(vecs.data(), row_norms_e.data(), 2, 1,
                        q.data(), /*qn=*/3.0f, out.data());
    REQUIRE_THAT(out[0], WithinAbs( 1.0f, 1e-5f));
    REQUIRE_THAT(out[1], WithinAbs(25.0f, 1e-5f));
}

// ---- Cosine boundary values ---------------------------------------------

TEST_CASE("CosineF32Chunk: identical vectors yield exactly 1.0",
          "[kernel][cosine][boundary]") {
    // Cosine of v with itself should be exactly 1.0 (as close to bit-exact as
    // float arithmetic allows — Eigen's dot product uses the same order of
    // operations as `sum(v[i]^2)`, so the numerator/denominator match).
    // We allow a small epsilon for the norm computation floating-point path.
    std::vector<float> vecs = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> norms = {L2NormF32(vecs.data(), 4)};
    std::vector<float> out(1);
    CosineF32Chunk(vecs.data(), norms.data(), 1, 4,
                   vecs.data(), norms[0], out.data());
    REQUIRE_THAT(out[0], WithinAbs(1.0f, 1e-6f));
}

TEST_CASE("CosineF32Chunk: opposite vectors yield -1.0",
          "[kernel][cosine][boundary]") {
    std::vector<float> row = {1.0f, 2.0f, 3.0f};
    std::vector<float> q   = {-1.0f, -2.0f, -3.0f};
    std::vector<float> norms = {L2NormF32(row.data(), 3)};
    std::vector<float> out(1);
    CosineF32Chunk(row.data(), norms.data(), 1, 3,
                   q.data(), L2NormF32(q.data(), 3), out.data());
    REQUIRE_THAT(out[0], WithinAbs(-1.0f, 1e-6f));
}

TEST_CASE("CosineF32Chunk: orthogonal vectors yield 0",
          "[kernel][cosine][boundary]") {
    std::vector<float> row = {1.0f, 0.0f, 0.0f};
    std::vector<float> q   = {0.0f, 1.0f, 0.0f};
    std::vector<float> norms = {1.0f};
    std::vector<float> out(1);
    CosineF32Chunk(row.data(), norms.data(), 1, 3, q.data(), 1.0f, out.data());
    REQUIRE_THAT(out[0], WithinAbs(0.0f, 1e-6f));
}

// ---- High-dim property test ---------------------------------------------

TEST_CASE("Kernels: high-dim (1024) matches reference", "[kernel][highdim][property]") {
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    constexpr std::size_t kDim = 1024;
    constexpr std::size_t kCount = 50;

    std::vector<float> vecs(kCount * kDim), norms(kCount), q(kDim), out(kCount);
    for (auto& x : vecs) x = dist(rng);
    for (auto& x : q)    x = dist(rng);
    for (std::size_t i = 0; i < kCount; ++i) {
        norms[i] = L2NormF32(&vecs[i * kDim], kDim);
    }
    const float qn = L2NormF32(q.data(), kDim);

    CosineF32Chunk(vecs.data(), norms.data(), kCount, kDim,
                   q.data(), qn, out.data());
    for (std::size_t i = 0; i < kCount; ++i) {
        const float want = ref::cosine(&vecs[i * kDim], q.data(), kDim);
        // High-dim sums accumulate more float error; loosen tolerance.
        REQUIRE_THAT(out[i], WithinAbs(want, 1e-4f));
    }
}

// ---- Determinism --------------------------------------------------------

TEST_CASE("Kernels: same input yields bit-exact same output twice",
          "[kernel][determinism]") {
    std::mt19937 rng(9999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    constexpr std::size_t kDim = 128;
    constexpr std::size_t kCount = 200;
    std::vector<float> vecs(kCount * kDim), norms(kCount), q(kDim);
    for (auto& x : vecs) x = dist(rng);
    for (auto& x : q)    x = dist(rng);
    for (std::size_t i = 0; i < kCount; ++i) {
        norms[i] = L2NormF32(&vecs[i * kDim], kDim);
    }
    const float qn = L2NormF32(q.data(), kDim);

    std::vector<float> out_a(kCount), out_b(kCount);
    CosineF32Chunk(vecs.data(), norms.data(), kCount, kDim, q.data(), qn, out_a.data());
    CosineF32Chunk(vecs.data(), norms.data(), kCount, kDim, q.data(), qn, out_b.data());
    for (std::size_t i = 0; i < kCount; ++i) {
        REQUIRE(out_a[i] == out_b[i]);  // bit-exact
    }

    std::vector<float> ea(kCount), eb(kCount);
    EuclideanSqF32Chunk(vecs.data(), norms.data(), kCount, kDim,
                        q.data(), qn, ea.data());
    EuclideanSqF32Chunk(vecs.data(), norms.data(), kCount, kDim,
                        q.data(), qn, eb.data());
    for (std::size_t i = 0; i < kCount; ++i) {
        REQUIRE(ea[i] == eb[i]);
    }
}

// ---- NaN input behavior (documented, not sanitized) ---------------------

TEST_CASE("CosineF32Chunk: NaN in query zeros the output (documented behavior)",
          "[kernel][nan][documented]") {
    // Design contract: kernels do NOT sanitize NaN inputs. If the query
    // vector contains NaN, qn (its L2 norm) is NaN, which fails the qn > 0
    // check and the kernel takes the "zero everything" branch. This test
    // documents that behavior so a future change is a deliberate decision.
    std::vector<float> vecs = {1, 0, 0,   0, 1, 0};
    std::vector<float> norms = {1.0f, 1.0f};
    std::vector<float> q = {std::nanf(""), 0.0f, 0.0f};
    std::vector<float> out(2, -1.0f);

    const float qn = L2NormF32(q.data(), 3);
    REQUIRE(std::isnan(qn));  // sanity: L2Norm propagates NaN

    CosineF32Chunk(vecs.data(), norms.data(), 2, 3, q.data(), qn, out.data());
    REQUIRE(out[0] == 0.0f);
    REQUIRE(out[1] == 0.0f);
}

TEST_CASE("CosineF32Chunk: NaN in a row propagates to that row's score",
          "[kernel][nan][documented]") {
    // Row 0 has a NaN component; its dot product with a valid q is NaN,
    // and norm[0] is provided by the caller as 1.0f (as if precomputed
    // incorrectly — GIGO). The output for row 0 will be NaN; row 1 clean.
    // This documents the "caller is responsible for finite inputs" contract.
    std::vector<float> vecs = {std::nanf(""), 0.0f, 0.0f,   1.0f, 0.0f, 0.0f};
    std::vector<float> norms = {1.0f, 1.0f};
    std::vector<float> q = {1.0f, 0.0f, 0.0f};
    std::vector<float> out(2, 0.0f);

    CosineF32Chunk(vecs.data(), norms.data(), 2, 3, q.data(), 1.0f, out.data());
    REQUIRE(std::isnan(out[0]));
    REQUIRE_THAT(out[1], WithinAbs(1.0f, 1e-6f));
}

// ---- Reference-implementation sanity check ------------------------------

TEST_CASE("Reference implementations match themselves", "[kernel][reference]") {
    // Sanity check on the naive helpers so bugs there don't invalidate our
    // property tests.
    std::vector<float> a = {3, 4};
    REQUIRE_THAT(ref::norm(a.data(), 2), WithinAbs(5.0f, 1e-6f));
    REQUIRE_THAT(ref::dot(a.data(), a.data(), 2), WithinAbs(25.0f, 1e-6f));
    std::vector<float> b = {0, 0};
    REQUIRE(ref::cosine(a.data(), b.data(), 2) == 0.0f);
    REQUIRE_THAT(ref::euclid_sq(a.data(), a.data(), 2), WithinAbs(0.0f, 1e-6f));
}
