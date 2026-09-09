#ifndef ONNX2NCNN_HPP
#define ONNX2NCNN_HPP

#include <onnx/onnx_pb.h>
#include <optional>
#include <string>
#include <tuple>

extern std::optional<std::tuple<char*, unsigned char*>> onnx2ncnn(ONNX_NAMESPACE::ModelProto& model);

#endif // ONNX2NCNN_HPP
