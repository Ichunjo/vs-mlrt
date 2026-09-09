#ifndef ONNX_UTILS_H
#define ONNX_UTILS_H

#include <cstdint>
#include <onnx/onnx_pb.h>
#include <string>
#include <string_view>
#include <variant>

std::variant<std::string, ONNX_NAMESPACE::ModelProto>
loadONNX(const std::string_view& path, int64_t tile_w, int64_t tile_h, bool path_is_serialization) noexcept;

#endif
