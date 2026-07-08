/*
 * Block codec implementation. See block_codec.h for the wire-format
 * documentation.
 */
#include "block_codec.h"

#include <cstring>

#include "chronosv/ring_buffer.h"   // dtype_size()

namespace chronosv::internal {

/* -------------------------------------------------------------------------- */
/* Little-endian read/write helpers.                                          */
/* -------------------------------------------------------------------------- */

/* On our supported platforms (macOS ARM64, Linux ARM64/x86_64) the CPU is
 * little-endian and unaligned memcpy is safe and single-instruction on
 * modern compilers. Using memcpy avoids UB from type-punned unaligned reads. */

static inline void put_u16_le(std::uint8_t* p, std::uint16_t v) noexcept {
    std::memcpy(p, &v, sizeof(v));
}
static inline void put_u32_le(std::uint8_t* p, std::uint32_t v) noexcept {
    std::memcpy(p, &v, sizeof(v));
}
static inline void put_i64_le(std::uint8_t* p, std::int64_t v) noexcept {
    std::memcpy(p, &v, sizeof(v));
}

static inline std::uint16_t get_u16_le(const std::uint8_t* p) noexcept {
    std::uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
static inline std::uint32_t get_u32_le(const std::uint8_t* p) noexcept {
    std::uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
static inline std::int64_t get_i64_le(const std::uint8_t* p) noexcept {
    std::int64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

/* -------------------------------------------------------------------------- */
/* CRC32/IEEE — table-driven, polynomial 0xEDB88320 (reflected 0x04C11DB7).   */
/* Matches zlib crc32() and Python zlib.crc32(). Chosen for cross-tool         */
/* verifiability, not raw speed — the alternative (CRC32C, Castagnoli) has     */
/* HW acceleration on modern CPUs but no such universal tool support.          */
/* -------------------------------------------------------------------------- */

namespace {
struct Crc32Table {
    std::uint32_t t[256];
    constexpr Crc32Table() : t{} {
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) {
                c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            }
            t[i] = c;
        }
    }
};
inline constexpr Crc32Table kCrcTable{};
}  // namespace

std::uint32_t crc32_ieee(const std::uint8_t* data, std::size_t len) noexcept {
    std::uint32_t c = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < len; ++i) {
        c = kCrcTable.t[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

/* -------------------------------------------------------------------------- */
/* Size computation                                                           */
/* -------------------------------------------------------------------------- */

std::size_t block_serialized_size(std::uint32_t count,
                                  std::uint32_t dim,
                                  std::uint32_t payload_size_bytes,
                                  chronosv_dtype_t dtype) noexcept {
    const std::size_t vsz = dtype_size(dtype);
    if (vsz == 0) return 0;  // unknown dtype

    std::size_t s = kBlockHeaderSize;
    s += static_cast<std::size_t>(count) * sizeof(std::int64_t);        // timestamps
    s += static_cast<std::size_t>(count) * dim * vsz;                    // vectors
    if (dtype == CHRONOSV_DTYPE_INT8) {
        s += static_cast<std::size_t>(count) * sizeof(float);            // scales
    }
    if (payload_size_bytes > 0) {
        s += static_cast<std::size_t>(count) * payload_size_bytes;       // payloads
    }
    s += kBlockCrcSize;
    return s;
}

std::size_t block_serialized_size(const Block& blk) noexcept {
    return block_serialized_size(blk.count, blk.dim,
                                 blk.payload_size_bytes, blk.dtype);
}

/* -------------------------------------------------------------------------- */
/* Serialize                                                                  */
/* -------------------------------------------------------------------------- */

std::size_t serialize_block(const Block& blk, std::uint8_t* out_bytes) noexcept {
    if (!out_bytes) return 0;
    if (blk.dim == 0) return 0;
    const std::size_t vsz = dtype_size(blk.dtype);
    if (vsz == 0) return 0;
    // count == 0 is valid (empty block, e.g. an eviction cycle with nothing
    // to persist). We still write a header + CRC to preserve structure.
    if (blk.count > 0) {
        if (!blk.timestamps || !blk.vectors) return 0;
        if (blk.dtype == CHRONOSV_DTYPE_INT8 && !blk.scales) return 0;
        if (blk.payload_size_bytes > 0 && !blk.payloads) return 0;
    }

    const std::uint16_t flags =
        (blk.payload_size_bytes > 0 ? kFlagHasPayload : 0) |
        (blk.dtype == CHRONOSV_DTYPE_INT8 ? kFlagIsInt8 : 0);

    std::uint8_t* p = out_bytes;

    /* Header (40 bytes) */
    put_u32_le(p +  0, kBlockMagic);
    put_u16_le(p +  4, kBlockVersion);
    put_u16_le(p +  6, flags);
    put_u32_le(p +  8, blk.dim);
    put_u32_le(p + 12, blk.count);
    put_i64_le(p + 16, blk.t_start_ms);
    put_i64_le(p + 24, blk.t_end_ms);
    put_u32_le(p + 32, blk.payload_size_bytes);
    put_u32_le(p + 36, 0);  // reserved
    p += kBlockHeaderSize;

    /* Timestamps */
    const std::size_t ts_bytes = static_cast<std::size_t>(blk.count) * sizeof(std::int64_t);
    if (ts_bytes > 0) {
        std::memcpy(p, blk.timestamps, ts_bytes);
        p += ts_bytes;
    }

    /* Vectors */
    const std::size_t vec_bytes = static_cast<std::size_t>(blk.count) * blk.dim * vsz;
    if (vec_bytes > 0) {
        std::memcpy(p, blk.vectors, vec_bytes);
        p += vec_bytes;
    }

    /* Scales (INT8 only) */
    if (blk.dtype == CHRONOSV_DTYPE_INT8) {
        const std::size_t sc_bytes = static_cast<std::size_t>(blk.count) * sizeof(float);
        if (sc_bytes > 0) {
            std::memcpy(p, blk.scales, sc_bytes);
            p += sc_bytes;
        }
    }

    /* Payloads (only if enabled) */
    if (blk.payload_size_bytes > 0) {
        const std::size_t pl_bytes =
            static_cast<std::size_t>(blk.count) * blk.payload_size_bytes;
        if (pl_bytes > 0) {
            std::memcpy(p, blk.payloads, pl_bytes);
            p += pl_bytes;
        }
    }

    /* CRC over everything above */
    const std::size_t body_bytes = static_cast<std::size_t>(p - out_bytes);
    const std::uint32_t crc = crc32_ieee(out_bytes, body_bytes);
    put_u32_le(p, crc);
    p += kBlockCrcSize;

    return static_cast<std::size_t>(p - out_bytes);
}

/* -------------------------------------------------------------------------- */
/* Decode                                                                     */
/* -------------------------------------------------------------------------- */

chronosv_error_t decode_block(const std::uint8_t* bytes,
                              std::size_t         len,
                              DecodedBlockHeader* out_hdr,
                              std::int64_t*       out_timestamps,
                              void*               out_vectors,
                              float*              out_scales,
                              std::uint8_t*       out_payloads) noexcept {
    /* INVALID_ARG: the CALLER did something wrong. */
    if (!bytes || !out_hdr) return CHRONOSV_ERR_INVALID_ARG;

    /* CORRUPTION: the BYTES are wrong. Anything past this point means the
     * data doesn't parse as a valid block, regardless of how it got here.
     * Storage backends translate this into on_corruption sink events. */
    if (len < kBlockHeaderSize + kBlockCrcSize) return CHRONOSV_ERR_CORRUPTION;

    /* Header parse (before CRC check — we need dim/count to compute expected
     * length). Values are still trusted only AFTER CRC verifies. */
    const std::uint32_t magic       = get_u32_le(bytes + 0);
    const std::uint16_t version     = get_u16_le(bytes + 4);
    const std::uint16_t flags       = get_u16_le(bytes + 6);
    const std::uint32_t dim         = get_u32_le(bytes + 8);
    const std::uint32_t count       = get_u32_le(bytes + 12);
    const std::int64_t  t_start     = get_i64_le(bytes + 16);
    const std::int64_t  t_end       = get_i64_le(bytes + 24);
    const std::uint32_t payload_sz  = get_u32_le(bytes + 32);
    /* bytes + 36 is reserved — no read */

    if (magic != kBlockMagic)      return CHRONOSV_ERR_CORRUPTION;
    if (version != kBlockVersion)  return CHRONOSV_ERR_CORRUPTION;

    const bool is_int8 = (flags & kFlagIsInt8) != 0;
    const bool has_pl  = (flags & kFlagHasPayload) != 0;
    /* Reject unknown flag bits — future formats must bump version, not
     * silently repurpose reserved bits. */
    if ((flags & ~(kFlagHasPayload | kFlagIsInt8)) != 0) return CHRONOSV_ERR_CORRUPTION;
    /* Payload flag must match payload_size_bytes state. */
    if (has_pl != (payload_sz > 0)) return CHRONOSV_ERR_CORRUPTION;

    const chronosv_dtype_t dtype =
        is_int8 ? CHRONOSV_DTYPE_INT8 : CHRONOSV_DTYPE_FLOAT32;
    const std::size_t vsz = dtype_size(dtype);
    if (vsz == 0 || dim == 0)     return CHRONOSV_ERR_CORRUPTION;

    /* Verify total size matches what the header says. */
    const std::size_t expected =
        block_serialized_size(count, dim, payload_sz, dtype);
    if (expected == 0 || len < expected)  return CHRONOSV_ERR_CORRUPTION;
    /* Note: len > expected is allowed (caller passed a larger buffer with
     * trailing junk). Only check that we have enough. */

    /* CRC covers everything except the trailing 4 bytes. */
    const std::size_t body_bytes = expected - kBlockCrcSize;
    const std::uint32_t stored_crc   = get_u32_le(bytes + body_bytes);
    const std::uint32_t computed_crc = crc32_ieee(bytes, body_bytes);
    if (stored_crc != computed_crc) return CHRONOSV_ERR_CORRUPTION;

    /* Publish header. */
    out_hdr->dim                = dim;
    out_hdr->count              = count;
    out_hdr->t_start_ms         = t_start;
    out_hdr->t_end_ms           = t_end;
    out_hdr->payload_size_bytes = payload_sz;
    out_hdr->dtype              = dtype;

    /* Extract data sections if requested. Walk the same offsets serialize
     * uses, but only touch out buffers when non-null. */
    const std::uint8_t* p = bytes + kBlockHeaderSize;

    const std::size_t ts_bytes = static_cast<std::size_t>(count) * sizeof(std::int64_t);
    if (out_timestamps && ts_bytes > 0) {
        std::memcpy(out_timestamps, p, ts_bytes);
    }
    p += ts_bytes;

    const std::size_t vec_bytes = static_cast<std::size_t>(count) * dim * vsz;
    if (out_vectors && vec_bytes > 0) {
        std::memcpy(out_vectors, p, vec_bytes);
    }
    p += vec_bytes;

    if (is_int8) {
        const std::size_t sc_bytes = static_cast<std::size_t>(count) * sizeof(float);
        if (out_scales && sc_bytes > 0) {
            std::memcpy(out_scales, p, sc_bytes);
        }
        p += sc_bytes;
    }

    if (payload_sz > 0) {
        const std::size_t pl_bytes =
            static_cast<std::size_t>(count) * payload_sz;
        if (out_payloads && pl_bytes > 0) {
            std::memcpy(out_payloads, p, pl_bytes);
        }
        p += pl_bytes;
    }

    return CHRONOSV_OK;
}

}  // namespace chronosv::internal
