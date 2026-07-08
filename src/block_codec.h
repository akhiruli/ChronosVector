/*
 * ChronosVector — Block codec (internal).
 *
 * Byte-level serialization / deserialization for the persisted block format
 * Pure: no I/O, no allocation on the hot
 * path, no dependency on RocksDB (deliberately — the codec is testable in
 * complete isolation from the storage backend).
 *
 * Wire format (little-endian throughout — matches all supported platforms;
 * BE hosts would require conversion, not currently supported):
 *
 *   Offset  Size            Field
 *   0       4               magic       = 'C''V''B''0' (0x30 0x42 0x56 0x43)
 *   4       2               version     = 1
 *   6       2               flags       bit 0: has_payload
 *                                       bit 1: is_int8
 *                                       (higher bits reserved, must be 0)
 *   8       4               dim
 *   12      4               count
 *   16      8               t_start_ms
 *   24      8               t_end_ms
 *   32      4               payload_size_bytes
 *   36      4               reserved (zeros)
 *   40      count*8         timestamps[count]
 *   ..      count*dim*Ds    vectors[count][dim]      (Ds = 4 for FLOAT32, 1 for INT8)
 *   ..      count*4         scales[count]            (only if flags.is_int8)
 *   ..      count*payload   payloads[count][payload_size_bytes] (only if flags.has_payload)
 *   tail    4               crc32/ieee of everything above
 *
 * The CRC catches encoder bugs and pre-storage memory flips; RocksDB's own
 * block CRC catches physical media corruption. Both layers stay (see
 */
#ifndef CHRONOSV_BLOCK_CODEC_H
#define CHRONOSV_BLOCK_CODEC_H

#include <cstddef>
#include <cstdint>

#include "chronosv/storage_backend.h"   // Block
#include "chronosv/types.h"              // chronosv_dtype_t, chronosv_error_t

namespace chronosv::internal {

/* Header field constants. Public so tests can assert byte offsets. */
inline constexpr std::uint32_t kBlockMagic       = 0x30425643u; /* 'C','V','B','0' in LE reads */
inline constexpr std::uint16_t kBlockVersion     = 1;
inline constexpr std::size_t   kBlockHeaderSize  = 40;
inline constexpr std::size_t   kBlockCrcSize     = 4;
inline constexpr std::uint16_t kFlagHasPayload   = 1u << 0;
inline constexpr std::uint16_t kFlagIsInt8       = 1u << 1;

/* Compute the exact byte size a serialized block will occupy given its
 * shape. Deterministic, no I/O. */
std::size_t block_serialized_size(std::uint32_t count,
                                  std::uint32_t dim,
                                  std::uint32_t payload_size_bytes,
                                  chronosv_dtype_t dtype) noexcept;

/* Convenience wrapper for a Block reference. */
std::size_t block_serialized_size(const Block& blk) noexcept;

/* Serialize a Block into out_bytes. out_bytes must have room for at least
 * block_serialized_size(blk) bytes; if it doesn't, behavior is undefined
 * (the codec cannot check on the caller's behalf).
 *
 * Returns the number of bytes actually written on success, or 0 on invalid
 * input (null pointers, dim==0, unknown dtype). */
std::size_t serialize_block(const Block& blk, std::uint8_t* out_bytes) noexcept;

/* Fields recovered from a decoded block header. Populated by decode_block()
 * before any data extraction. */
struct DecodedBlockHeader {
    std::uint32_t    dim;
    std::uint32_t    count;
    std::int64_t     t_start_ms;
    std::int64_t     t_end_ms;
    std::uint32_t    payload_size_bytes;   /* 0 if no payload */
    chronosv_dtype_t dtype;
};

/* Decode a serialized block. Verifies magic, version, CRC in that order.
 *
 * If a given `out_*` data pointer is nullptr, that section is not extracted
 * (but the CRC still covers it, so integrity is still verified). Passing
 * ALL out pointers nullptr except out_hdr yields a "header-only" inspection
 * useful for the chronosv_open recovery scan.
 *
 * Caller is responsible for sizing out buffers per DecodedBlockHeader:
 *   out_timestamps: at least header.count entries
 *   out_vectors:    at least header.count * header.dim * dtype_size bytes
 *   out_scales:     at least header.count entries (only used if is_int8)
 *   out_payloads:   at least header.count * header.payload_size_bytes bytes
 *
 * Returns:
 *   CHRONOSV_OK              on success
 *   CHRONOSV_ERR_INVALID_ARG the CALLER passed a bad argument — `bytes` or
 *                            `out_hdr` is null. Recoverable by the caller.
 *   CHRONOSV_ERR_CORRUPTION  the BYTES are wrong — truncated buffer,
 *                            bad magic, bad version, unknown flag bits,
 *                            payload flag / size mismatch, dim==0,
 *                            unknown dtype, size doesn't match the header,
 *                            or CRC mismatch. The storage backend fires
 *                            an on_corruption sink event when it sees
 *                            this from persisted data. */
chronosv_error_t decode_block(const std::uint8_t* bytes,
                              std::size_t         len,
                              DecodedBlockHeader* out_hdr,
                              std::int64_t*       out_timestamps,
                              void*               out_vectors,
                              float*              out_scales,
                              std::uint8_t*       out_payloads) noexcept;

/* Compute the CRC32/IEEE of `len` bytes at `data`. Exposed so tests can
 * verify externally and so future users (RocksDB backend health checks
 * etc.) don't have to reimplement. Table-driven, no allocation. */
std::uint32_t crc32_ieee(const std::uint8_t* data, std::size_t len) noexcept;

}  // namespace chronosv::internal

#endif  // CHRONOSV_BLOCK_CODEC_H
