/*
 * Block codec unit tests. Focus:
 *   1. Round-trip (serialize → decode) preserves every field bit-exactly.
 *   2. Byte-level layout (byte offsets, values).
 *   3. CRC32 catches single-bit flips anywhere in the block.
 *   4. Magic / version / unknown-flag / truncated inputs are rejected.
 *   5. Empty (count=0) blocks round-trip.
 *   6. Both dtype variants (FLOAT32 default; INT8 exercised even though the
 *      kernel path is Phase 1.5 — the codec must handle it now).
 *   7. Payload with and without.
 *
 * The codec is pure and has no I/O dependency, so these tests can run in
 * complete isolation from RocksDB or the engine.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cstdint>
#include <cstring>
#include <random>
#include <string_view>
#include <vector>

#include "block_codec.h"
#include "chronosv/storage_backend.h"
#include "chronosv/types.h"

using chronosv::internal::Block;
using chronosv::internal::DecodedBlockHeader;
using chronosv::internal::block_serialized_size;
using chronosv::internal::crc32_ieee;
using chronosv::internal::decode_block;
using chronosv::internal::serialize_block;
using chronosv::internal::kBlockCrcSize;
using chronosv::internal::kBlockHeaderSize;
using chronosv::internal::kBlockMagic;
using chronosv::internal::kBlockVersion;
using chronosv::internal::kFlagHasPayload;
using chronosv::internal::kFlagIsInt8;

/* ==========================================================================
 * CRC32 self-check (independent of block codec)
 * ========================================================================== */

TEST_CASE("crc32_ieee: known vectors match zlib", "[block_codec][crc]") {
    // Standard test vector for CRC-32/IEEE (matches zlib.crc32 in Python):
    //   crc32(b"") = 0
    //   crc32(b"a") = 0xE8B7BE43
    //   crc32(b"123456789") = 0xCBF43926
    REQUIRE(crc32_ieee(nullptr, 0) == 0u);
    const std::uint8_t a[] = {'a'};
    REQUIRE(crc32_ieee(a, 1) == 0xE8B7BE43u);
    const std::uint8_t s[] = {'1','2','3','4','5','6','7','8','9'};
    REQUIRE(crc32_ieee(s, sizeof(s)) == 0xCBF43926u);
}

TEST_CASE("crc32_ieee: single-bit flip changes the checksum",
          "[block_codec][crc]") {
    std::vector<std::uint8_t> a(256), b(256);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = b[i] = static_cast<std::uint8_t>(i);
    b[100] ^= 0x01;   // flip one bit
    REQUIRE(crc32_ieee(a.data(), a.size()) != crc32_ieee(b.data(), b.size()));
}

/* ==========================================================================
 * Serialized-size accounting
 * ========================================================================== */

TEST_CASE("block_serialized_size: matches spec for canonical shapes",
          "[block_codec][size]") {
    // FLOAT32, count=100, dim=128, no payload:
    //   header(40) + ts(100*8) + vec(100*128*4) + crc(4)
    //   = 40 + 800 + 51200 + 4 = 52044
    REQUIRE(block_serialized_size(100, 128, 0, CHRONOSV_DTYPE_FLOAT32) == 52044u);

    // INT8, count=100, dim=128, payload=16:
    //   header(40) + ts(800) + vec(100*128*1) + scales(100*4) + payload(100*16) + crc(4)
    //   = 40 + 800 + 12800 + 400 + 1600 + 4 = 15644
    REQUIRE(block_serialized_size(100, 128, 16, CHRONOSV_DTYPE_INT8) == 15644u);

    // Empty block (count=0): just header + crc = 44
    REQUIRE(block_serialized_size(0, 128, 0, CHRONOSV_DTYPE_FLOAT32) == 44u);
}

/* ==========================================================================
 * Header layout — byte-level checks against the on-disk spec
 * ========================================================================== */

TEST_CASE("Serialized header: byte-level layout matches spec",
          "[block_codec][layout]") {
    std::vector<std::int64_t> ts = {1000, 2000, 3000};
    std::vector<float> vecs = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
    };
    Block blk{};
    blk.sensor_id          = "s";
    blk.block_id           = 42;   // not persisted — RocksDB key holds it
    blk.count              = 3;
    blk.dim                = 4;
    blk.dtype              = CHRONOSV_DTYPE_FLOAT32;
    blk.payload_size_bytes = 0;
    blk.t_start_ms         = 1000;
    blk.t_end_ms           = 3000;
    blk.timestamps         = ts.data();
    blk.vectors            = vecs.data();

    std::vector<std::uint8_t> buf(block_serialized_size(blk));
    const std::size_t written = serialize_block(blk, buf.data());
    REQUIRE(written == buf.size());

    // magic at offset 0..3 = 'C','V','B','0'
    REQUIRE(buf[0] == 'C');
    REQUIRE(buf[1] == 'V');
    REQUIRE(buf[2] == 'B');
    REQUIRE(buf[3] == '0');

    // version at 4..5 = 1
    REQUIRE(buf[4] == 1);
    REQUIRE(buf[5] == 0);

    // flags at 6..7 = 0 (no payload, not int8)
    REQUIRE(buf[6] == 0);
    REQUIRE(buf[7] == 0);

    // dim at 8..11 = 4 (little-endian)
    REQUIRE(buf[8]  == 4);
    REQUIRE(buf[9]  == 0);
    REQUIRE(buf[10] == 0);
    REQUIRE(buf[11] == 0);

    // count at 12..15 = 3 (little-endian)
    REQUIRE(buf[12] == 3);
    REQUIRE(buf[13] == 0);

    // reserved at 36..39 = zeros
    REQUIRE(buf[36] == 0);
    REQUIRE(buf[37] == 0);
    REQUIRE(buf[38] == 0);
    REQUIRE(buf[39] == 0);
}

/* ==========================================================================
 * Round-trip: FLOAT32
 * ========================================================================== */

TEST_CASE("Round-trip: FLOAT32 vectors, no payload", "[block_codec][roundtrip]") {
    constexpr std::uint32_t kCount = 50;
    constexpr std::uint32_t kDim   = 128;

    // Deterministic data.
    std::vector<std::int64_t> ts(kCount);
    std::vector<float>        vecs(kCount * kDim);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (std::uint32_t i = 0; i < kCount; ++i) ts[i] = 1000 + i * 10;
    for (auto& x : vecs) x = dist(rng);

    Block blk{};
    blk.sensor_id  = "sensor_a";
    blk.block_id   = 7;
    blk.count      = kCount;
    blk.dim        = kDim;
    blk.dtype      = CHRONOSV_DTYPE_FLOAT32;
    blk.t_start_ms = ts.front();
    blk.t_end_ms   = ts.back();
    blk.timestamps = ts.data();
    blk.vectors    = vecs.data();

    std::vector<std::uint8_t> buf(block_serialized_size(blk));
    REQUIRE(serialize_block(blk, buf.data()) == buf.size());

    // Decode with all out buffers provided.
    DecodedBlockHeader hdr{};
    std::vector<std::int64_t> ts_out(kCount);
    std::vector<float>        vecs_out(kCount * kDim);
    REQUIRE(decode_block(buf.data(), buf.size(),
                         &hdr,
                         ts_out.data(),
                         vecs_out.data(),
                         nullptr, nullptr) == CHRONOSV_OK);
    REQUIRE(hdr.dim == kDim);
    REQUIRE(hdr.count == kCount);
    REQUIRE(hdr.dtype == CHRONOSV_DTYPE_FLOAT32);
    REQUIRE(hdr.payload_size_bytes == 0);
    REQUIRE(hdr.t_start_ms == ts.front());
    REQUIRE(hdr.t_end_ms   == ts.back());
    REQUIRE(ts_out == ts);
    REQUIRE(std::memcmp(vecs_out.data(), vecs.data(), vecs.size() * sizeof(float)) == 0);
}

/* ==========================================================================
 * Round-trip: INT8 with scales and payload
 * ========================================================================== */

TEST_CASE("Round-trip: INT8 vectors + scales + payload",
          "[block_codec][roundtrip][int8][payload]") {
    constexpr std::uint32_t kCount   = 10;
    constexpr std::uint32_t kDim     = 8;
    constexpr std::uint32_t kPayload = 16;

    std::vector<std::int64_t> ts(kCount);
    std::vector<std::int8_t>  vecs(kCount * kDim);
    std::vector<float>        scales(kCount);
    std::vector<std::uint8_t> payloads(kCount * kPayload);
    for (std::uint32_t i = 0; i < kCount; ++i) {
        ts[i]     = 500 + i;
        scales[i] = 1.0f + 0.1f * i;
        for (std::uint32_t d = 0; d < kDim; ++d) {
            vecs[i * kDim + d] = static_cast<std::int8_t>((i * 7 + d) - 64);
        }
        for (std::uint32_t k = 0; k < kPayload; ++k) {
            payloads[i * kPayload + k] = static_cast<std::uint8_t>((i * 31 + k) & 0xFFu);
        }
    }

    Block blk{};
    blk.sensor_id          = "s";
    blk.block_id           = 0;
    blk.count              = kCount;
    blk.dim                = kDim;
    blk.dtype              = CHRONOSV_DTYPE_INT8;
    blk.payload_size_bytes = kPayload;
    blk.t_start_ms         = ts.front();
    blk.t_end_ms           = ts.back();
    blk.timestamps         = ts.data();
    blk.vectors            = vecs.data();
    blk.scales             = scales.data();
    blk.payloads           = payloads.data();

    std::vector<std::uint8_t> buf(block_serialized_size(blk));
    REQUIRE(serialize_block(blk, buf.data()) == buf.size());

    DecodedBlockHeader hdr{};
    std::vector<std::int64_t> ts_out(kCount);
    std::vector<std::int8_t>  vecs_out(kCount * kDim);
    std::vector<float>        scales_out(kCount);
    std::vector<std::uint8_t> payloads_out(kCount * kPayload);
    REQUIRE(decode_block(buf.data(), buf.size(),
                         &hdr,
                         ts_out.data(),
                         vecs_out.data(),
                         scales_out.data(),
                         payloads_out.data()) == CHRONOSV_OK);
    REQUIRE(hdr.dim == kDim);
    REQUIRE(hdr.count == kCount);
    REQUIRE(hdr.dtype == CHRONOSV_DTYPE_INT8);
    REQUIRE(hdr.payload_size_bytes == kPayload);
    REQUIRE(ts_out       == ts);
    REQUIRE(vecs_out     == vecs);
    REQUIRE(scales_out   == scales);
    REQUIRE(payloads_out == payloads);
}

/* ==========================================================================
 * Header-only inspection (all out data pointers nullptr)
 * ========================================================================== */

TEST_CASE("Decode with nullptr out buffers: header only, CRC still verified",
          "[block_codec][peek]") {
    // Recovery scan path — we want to know dim/count/timestamps-range
    // without paying the memcpy of the vector data.
    std::vector<std::int64_t> ts = {100, 200, 300};
    std::vector<float>        vecs(3 * 4, 0.5f);
    Block blk{};
    blk.count = 3; blk.dim = 4; blk.dtype = CHRONOSV_DTYPE_FLOAT32;
    blk.t_start_ms = 100; blk.t_end_ms = 300;
    blk.timestamps = ts.data(); blk.vectors = vecs.data();

    std::vector<std::uint8_t> buf(block_serialized_size(blk));
    serialize_block(blk, buf.data());

    DecodedBlockHeader hdr{};
    REQUIRE(decode_block(buf.data(), buf.size(),
                         &hdr,
                         nullptr, nullptr, nullptr, nullptr) == CHRONOSV_OK);
    REQUIRE(hdr.count == 3);
    REQUIRE(hdr.dim == 4);
    REQUIRE(hdr.t_start_ms == 100);
    REQUIRE(hdr.t_end_ms == 300);
}

/* ==========================================================================
 * Empty block (count=0) round-trips
 * ========================================================================== */

TEST_CASE("Round-trip: empty block (count=0)",
          "[block_codec][roundtrip][edge]") {
    Block blk{};
    blk.count = 0; blk.dim = 4; blk.dtype = CHRONOSV_DTYPE_FLOAT32;
    blk.t_start_ms = 0; blk.t_end_ms = 0;

    std::vector<std::uint8_t> buf(block_serialized_size(blk));
    REQUIRE(buf.size() == 44u);  // 40 header + 4 crc
    REQUIRE(serialize_block(blk, buf.data()) == 44u);

    DecodedBlockHeader hdr{};
    REQUIRE(decode_block(buf.data(), buf.size(),
                         &hdr, nullptr, nullptr, nullptr, nullptr) == CHRONOSV_OK);
    REQUIRE(hdr.count == 0);
    REQUIRE(hdr.dim == 4);
}

/* ==========================================================================
 * Corruption rejection
 * ========================================================================== */

namespace {
// Helper: build a valid serialized block for corruption tests.
std::vector<std::uint8_t> make_valid_block() {
    std::vector<std::int64_t> ts = {1, 2, 3};
    std::vector<float>        vecs(3 * 4, 1.5f);
    Block blk{};
    blk.count = 3; blk.dim = 4; blk.dtype = CHRONOSV_DTYPE_FLOAT32;
    blk.t_start_ms = 1; blk.t_end_ms = 3;
    blk.timestamps = ts.data(); blk.vectors = vecs.data();
    std::vector<std::uint8_t> buf(block_serialized_size(blk));
    serialize_block(blk, buf.data());
    return buf;
}
}  // namespace

TEST_CASE("Decode rejects: null bytes / null out_hdr is INVALID_ARG",
          "[block_codec][reject]") {
    // These are caller errors, not data corruption.
    auto buf = make_valid_block();
    DecodedBlockHeader hdr{};
    REQUIRE(decode_block(nullptr, buf.size(),
                         &hdr, nullptr, nullptr, nullptr, nullptr)
            == CHRONOSV_ERR_INVALID_ARG);
    REQUIRE(decode_block(buf.data(), buf.size(),
                         nullptr, nullptr, nullptr, nullptr, nullptr)
            == CHRONOSV_ERR_INVALID_ARG);
}

TEST_CASE("Decode rejects: truncated input is CORRUPTION",
          "[block_codec][reject][corruption]") {
    auto buf = make_valid_block();
    DecodedBlockHeader hdr{};

    // len < header+crc.
    REQUIRE(decode_block(buf.data(), 10,
                         &hdr, nullptr, nullptr, nullptr, nullptr)
            == CHRONOSV_ERR_CORRUPTION);

    // len drops one byte from the tail (drops part of CRC).
    REQUIRE(decode_block(buf.data(), buf.size() - 1,
                         &hdr, nullptr, nullptr, nullptr, nullptr)
            == CHRONOSV_ERR_CORRUPTION);
}

TEST_CASE("Decode rejects: bad magic is CORRUPTION",
          "[block_codec][reject][corruption]") {
    auto buf = make_valid_block();
    buf[0] = 'X';  // corrupt magic
    DecodedBlockHeader hdr{};
    REQUIRE(decode_block(buf.data(), buf.size(),
                         &hdr, nullptr, nullptr, nullptr, nullptr)
            == CHRONOSV_ERR_CORRUPTION);
}

TEST_CASE("Decode rejects: bad version is CORRUPTION",
          "[block_codec][reject][corruption]") {
    auto buf = make_valid_block();
    buf[4] = 99;   // corrupt version LSB
    DecodedBlockHeader hdr{};
    REQUIRE(decode_block(buf.data(), buf.size(),
                         &hdr, nullptr, nullptr, nullptr, nullptr)
            == CHRONOSV_ERR_CORRUPTION);
}

TEST_CASE("Decode rejects: unknown flag bits is CORRUPTION",
          "[block_codec][reject][corruption]") {
    auto buf = make_valid_block();
    buf[6] = 0x80;   // set a reserved bit
    DecodedBlockHeader hdr{};
    REQUIRE(decode_block(buf.data(), buf.size(),
                         &hdr, nullptr, nullptr, nullptr, nullptr)
            == CHRONOSV_ERR_CORRUPTION);
}

TEST_CASE("Decode rejects: CRC mismatch on any single-bit flip is CORRUPTION",
          "[block_codec][reject][corruption][crc]") {
    auto buf = make_valid_block();
    // Flip one bit in the middle of the vector data.
    const std::size_t mid = kBlockHeaderSize + 8 * 3 + 4;  // past ts, into vectors
    buf[mid] ^= 0x01;
    DecodedBlockHeader hdr{};
    REQUIRE(decode_block(buf.data(), buf.size(),
                         &hdr, nullptr, nullptr, nullptr, nullptr)
            == CHRONOSV_ERR_CORRUPTION);
}

TEST_CASE("Decode rejects: CRC mismatch on flipped CRC bytes is CORRUPTION",
          "[block_codec][reject][corruption][crc]") {
    auto buf = make_valid_block();
    buf[buf.size() - 1] ^= 0xFF;   // corrupt CRC itself
    DecodedBlockHeader hdr{};
    REQUIRE(decode_block(buf.data(), buf.size(),
                         &hdr, nullptr, nullptr, nullptr, nullptr)
            == CHRONOSV_ERR_CORRUPTION);
}

/* ==========================================================================
 * serialize_block parameter validation
 * ========================================================================== */

TEST_CASE("serialize_block rejects: null out buffer", "[block_codec][reject]") {
    Block blk{};
    blk.count = 1; blk.dim = 4; blk.dtype = CHRONOSV_DTYPE_FLOAT32;
    std::int64_t ts = 0; float vec[4] = {};
    blk.timestamps = &ts; blk.vectors = vec;
    REQUIRE(serialize_block(blk, nullptr) == 0u);
}

TEST_CASE("serialize_block rejects: dim=0", "[block_codec][reject]") {
    Block blk{};
    blk.count = 1; blk.dim = 0; blk.dtype = CHRONOSV_DTYPE_FLOAT32;
    std::uint8_t buf[100] = {};
    REQUIRE(serialize_block(blk, buf) == 0u);
}

TEST_CASE("serialize_block rejects: non-zero count with null data",
          "[block_codec][reject]") {
    Block blk{};
    blk.count = 1; blk.dim = 4; blk.dtype = CHRONOSV_DTYPE_FLOAT32;
    // timestamps and vectors are nullptr — invalid for count > 0.
    std::uint8_t buf[100] = {};
    REQUIRE(serialize_block(blk, buf) == 0u);
}
