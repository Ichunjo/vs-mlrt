#ifndef VSORT_INFERENCE_HELPER_H_
#define VSORT_INFERENCE_HELPER_H_

#include <VSHelper4.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <onnxruntime_c_api.h>
#include <optional>
#include <string>
#include <vector>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

using namespace std::string_literals;

struct TileDesc {
    int x;
    int y;
    int x_crop_start;
    int x_crop_end;
    int y_crop_start;
    int y_crop_end;
};

inline std::vector<TileDesc>
generateTiles(int width, int height, int tile_w, int tile_h, int overlap_w, int overlap_h) {
    int step_w = tile_w - 2 * overlap_w;
    int step_h = tile_h - 2 * overlap_h;

    std::vector<TileDesc> tiles;
    int y = 0;
    while (true) {
        int y_crop_start = (y == 0) ? 0 : overlap_h;
        int y_crop_end = (y == height - tile_h) ? 0 : overlap_h;

        int x = 0;
        while (true) {
            int x_crop_start = (x == 0) ? 0 : overlap_w;
            int x_crop_end = (x == width - tile_w) ? 0 : overlap_w;

            tiles.push_back(TileDesc{x, y, x_crop_start, x_crop_end, y_crop_start, y_crop_end});

            if (x + tile_w == width) {
                break;
            }
            x = std::min(x + step_w, width - tile_w);
        }

        if (y + tile_h == height) {
            break;
        }
        y = std::min(y + step_h, height - tile_h);
    }

    return tiles;
}

#endif // VSORT_INFERENCE_HELPER_H_
