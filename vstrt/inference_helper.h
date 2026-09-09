#ifndef VSTRT_INFERENCE_HELPER_H_
#define VSTRT_INFERENCE_HELPER_H_

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <VSHelper4.h>

#include "cuda_helper.h"
#include "trt_utils.h"

struct InputInfo {
    int width;
    int height;
    ptrdiff_t pitch;
    int bytes_per_sample;
    int tile_w;
    int tile_h;
};

struct OutputInfo {
    ptrdiff_t pitch;
    int bytes_per_sample;
};

struct IOInfo {
    InputInfo in;
    OutputInfo out;
    int w_scale;
    int h_scale;
    int overlap_w;
    int overlap_h;
};

static inline
std::optional<ErrorMessage> inference(
    const InferenceInstance & instance,
    int device_id,
    bool use_cuda_graph, 
    const IOInfo & info,
    const std::vector<const uint8_t *> & src_ptrs,
    const std::vector<uint8_t *> & dst_ptrs
) noexcept {

    const auto set_error = [](const ErrorMessage & error_message) {
        return error_message;
    };

    checkError(cudaSetDevice(device_id));

    int src_tile_w_bytes = info.in.tile_w * info.in.bytes_per_sample;
    int src_tile_bytes = info.in.tile_h * info.in.tile_w * info.in.bytes_per_sample;
    int dst_tile_w = info.in.tile_w * info.w_scale;
    int dst_tile_h = info.in.tile_h * info.h_scale;
    int dst_tile_w_bytes = dst_tile_w * info.out.bytes_per_sample;
    int dst_tile_bytes = dst_tile_h * dst_tile_w * info.out.bytes_per_sample;

    int step_w = info.in.tile_w - 2 * info.overlap_w;
    int step_h = info.in.tile_h - 2 * info.overlap_h;

    struct TileDesc {
        int x;
        int y;
        int x_crop_start;
        int x_crop_end;
        int y_crop_start;
        int y_crop_end;
    };

    std::vector<TileDesc> tiles;
    int y = 0;
    while (true) {
        int y_crop_start = (y == 0) ? 0 : info.overlap_h;
        int y_crop_end = (y == info.in.height - info.in.tile_h) ? 0 : info.overlap_h;

        int x = 0;
        while (true) {
            int x_crop_start = (x == 0) ? 0 : info.overlap_w;
            int x_crop_end = (x == info.in.width - info.in.tile_w) ? 0 : info.overlap_w;

            tiles.push_back(TileDesc{ x, y, x_crop_start, x_crop_end, y_crop_start, y_crop_end });

            if (x + info.in.tile_w == info.in.width) {
                break;
            }
            x = std::min(x + step_w, info.in.width - info.in.tile_w);
        }

        if (y + info.in.tile_h == info.in.height) {
            break;
        }
        y = std::min(y + step_h, info.in.height - info.in.tile_h);
    }

    if (tiles.empty()) {
        return {};
    }

    auto pack_tile = [&](size_t tile_idx, int b) -> void {
        const auto & tile = tiles[tile_idx];
        uint8_t * h_data = instance.src[b].h_data.data;
        for (const uint8_t * _src_ptr : src_ptrs) {
            const uint8_t * src_ptr { _src_ptr +
                tile.y * info.in.pitch + tile.x * info.in.bytes_per_sample
            };

            vsh::bitblt(
                h_data, src_tile_w_bytes,
                src_ptr, info.in.pitch,
                static_cast<size_t>(src_tile_w_bytes),
                static_cast<size_t>(info.in.tile_h)
            );

            h_data += src_tile_bytes;
        }
    };

    auto launch_tile = [&](int b) -> std::optional<ErrorMessage> {
        // 1. Host-to-Device transfer on h2d_stream
        checkError(cudaMemcpyAsync(
            instance.src[b].d_data, instance.src[b].h_data, instance.src[b].size,
            cudaMemcpyHostToDevice, instance.h2d_stream
        ));
        checkError(cudaEventRecord(instance.h2d_done[b], instance.h2d_stream));

        // 2. Compute stream waits for H2D
        checkError(cudaStreamWaitEvent(instance.stream, instance.h2d_done[b], 0));

        // 3. Launch TRT compute on compute stream (instance.stream)
        if (use_cuda_graph) {
            checkError(cudaGraphLaunch(instance.graphexec[b], instance.stream));
        } else {
            auto result = enqueueCompute(
                instance.src[b], instance.dst[b],
                instance.exec_context, instance.stream
            );
            if (result.has_value()) {
                return set_error(result.value());
            }
        }
        checkError(cudaEventRecord(instance.compute_done[b], instance.stream));

        // 4. D2H stream waits for compute
        checkError(cudaStreamWaitEvent(instance.d2h_stream, instance.compute_done[b], 0));

        // 5. Device-to-Host transfer on d2h_stream
        checkError(cudaMemcpyAsync(
            instance.dst[b].h_data, instance.dst[b].d_data, instance.dst[b].size,
            cudaMemcpyDeviceToHost, instance.d2h_stream
        ));
        checkError(cudaEventRecord(instance.d2h_done[b], instance.d2h_stream));

        return {};
    };

    auto unpack_tile = [&](size_t tile_idx, int b) -> std::optional<ErrorMessage> {
        const auto & tile = tiles[tile_idx];
        checkError(cudaEventSynchronize(instance.d2h_done[b]));

        const uint8_t * h_data = instance.dst[b].h_data.data;
        for (uint8_t * _dst_ptr : dst_ptrs) {
            uint8_t * dst_ptr { _dst_ptr +
                info.h_scale * tile.y * info.out.pitch + info.w_scale * tile.x * info.out.bytes_per_sample
            };

            vsh::bitblt(
                dst_ptr + (tile.y_crop_start * info.out.pitch + tile.x_crop_start * info.out.bytes_per_sample),
                info.out.pitch,
                h_data + (tile.y_crop_start * dst_tile_w_bytes + tile.x_crop_start * info.out.bytes_per_sample),
                dst_tile_w_bytes,
                static_cast<size_t>(dst_tile_w_bytes - (tile.x_crop_start + tile.x_crop_end) * info.out.bytes_per_sample),
                static_cast<size_t>(dst_tile_h - (tile.y_crop_start + tile.y_crop_end))
            );

            h_data += dst_tile_bytes;
        }

        return {};
    };

    for (size_t i = 0; i < tiles.size(); ++i) {
        int b = static_cast<int>(i % kNumBuffers);
        if (i >= kNumBuffers) {
            if (auto err = unpack_tile(i - kNumBuffers, b); err.has_value()) {
                return err;
            }
        }
        pack_tile(i, b);
        if (auto err = launch_tile(b); err.has_value()) {
            return err;
        }
    }

    // Drain remaining in-flight tiles
    size_t in_flight = std::min(tiles.size(), kNumBuffers);
    for (size_t k = in_flight; k > 0; --k) {
        size_t tile_idx = tiles.size() - k;
        int b = static_cast<int>(tile_idx % kNumBuffers);
        if (auto err = unpack_tile(tile_idx, b); err.has_value()) {
            return err;
        }
    }

    return {};
}

#endif // VSTRT_INFERENCE_HELPER_H_
