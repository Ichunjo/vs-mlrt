#ifndef VSTRT_TRT_UTILS_H_
#define VSTRT_TRT_UTILS_H_

#include <array>
#include <cstdint>
#include <memory>
#include <iostream>
#include <optional>
#include <string>
#include <variant>

#include <cuda_runtime.h>
#include <NvInferRuntime.h>

#include <VapourSynth4.h>

#include "cuda_helper.h"
#include "cuda_utils.h"

using ErrorMessage = std::string;

struct RequestedTileSize {
    int tile_w;
    int tile_h;
};

struct VideoSize {
    int width;
    int height;
};

using TileSize = std::variant<RequestedTileSize, VideoSize>;

constexpr size_t kNumBuffers = 3;

struct InferenceInstance {
    std::array<MemoryResource, kNumBuffers> src;
    std::array<MemoryResource, kNumBuffers> dst;
    StreamResource stream;
    StreamResource h2d_stream;
    StreamResource d2h_stream;
    std::array<EventResource, kNumBuffers> h2d_done;
    std::array<EventResource, kNumBuffers> compute_done;
    std::array<EventResource, kNumBuffers> d2h_done;
    std::unique_ptr<nvinfer1::IExecutionContext> exec_context;
    std::array<GraphExecResource, kNumBuffers> graphexec;
    Resource<uint8_t *, cudaFree> d_context_allocation;
};

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= verbosity) {
            if (vsapi && core) {
                int msgType;
                switch (severity) {
                    case Severity::kINTERNAL_ERROR:
                    case Severity::kERROR:
                        msgType = mtCritical;
                        break;
                    case Severity::kWARNING:
                        msgType = mtWarning;
                        break;
                    case Severity::kINFO:
                        msgType = mtInformation;
                        break;
                    default: // kVERBOSE
                        msgType = mtDebug;
                        break;
                }
                vsapi->logMessage(msgType, message, core);
            } else {
                std::cerr << message << '\n';
            }
        }
    }

public:
    Logger() = default;

    void set_verbosity(Severity value) noexcept {
        this->verbosity = value;
    }

    void set_vs_api(const VSAPI *api, VSCore *c) noexcept {
        this->vsapi = api;
        this->core = c;
    }

private:
    Severity verbosity;
    const VSAPI *vsapi = nullptr;
    VSCore *core = nullptr;
};

static inline
std::optional<int> selectProfile(
    const std::unique_ptr<nvinfer1::ICudaEngine> & engine,
    const TileSize & tile_size,
    int batch_size = 1
) noexcept {

    int tile_w, tile_h;
    if (std::holds_alternative<RequestedTileSize>(tile_size)) {
        tile_w = std::get<RequestedTileSize>(tile_size).tile_w;
        tile_h = std::get<RequestedTileSize>(tile_size).tile_h;
    } else {
        tile_w = std::get<VideoSize>(tile_size).width;
        tile_h = std::get<VideoSize>(tile_size).height;
    }

    auto input_name = engine->getIOTensorName(0);

    // finds the optimal profile
    for (int i = 0; i < engine->getNbOptimizationProfiles(); ++i) {
        nvinfer1::Dims opt_dims = engine->getProfileShape(
            input_name, i, nvinfer1::OptProfileSelector::kOPT
        );

        if (opt_dims.d[0] != batch_size) {
            continue;
        }
        if (opt_dims.d[2] == tile_h && opt_dims.d[3] == tile_w) {
            return i;
        }
    }

    // finds the first eligible profile
    for (int i = 0; i < engine->getNbOptimizationProfiles(); ++i) {
        nvinfer1::Dims min_dims = engine->getProfileShape(
            input_name, i, nvinfer1::OptProfileSelector::kMIN
        );

        if (min_dims.d[0] > batch_size) {
            continue;
        }
        if (min_dims.d[2] > tile_h || min_dims.d[3] > tile_w) {
            continue;
        }

        nvinfer1::Dims max_dims = engine->getProfileShape(
            input_name, i, nvinfer1::OptProfileSelector::kMAX
        );

        if (max_dims.d[0] < batch_size) {
            continue;
        }
        if (max_dims.d[2] < tile_h || max_dims.d[3] < tile_w) {
            continue;
        }

        return i;
    }

    // returns not-found
    return {};
}

static inline
std::optional<ErrorMessage> enqueueCompute(
    const MemoryResource & src,
    const MemoryResource & dst,
    const std::unique_ptr<nvinfer1::IExecutionContext> & exec_context,
    cudaStream_t stream
) noexcept {

    const auto set_error = [](const ErrorMessage & message) {
        return message;
    };

    auto input_name = exec_context->getEngine().getIOTensorName(0);
    auto output_name = exec_context->getEngine().getIOTensorName(1);

    if (!exec_context->setTensorAddress(input_name, src.d_data.data)) {
        return set_error("set input tensor address failed");
    }
    if (!exec_context->setTensorAddress(output_name, dst.d_data.data)) {
        return set_error("set output tensor address failed");
    }
    if (!exec_context->enqueueV3(stream)) {
        return set_error("enqueue error");
    }

    return {};
}

static inline
std::variant<ErrorMessage, GraphExecResource> getGraphExec(
    const MemoryResource & src, const MemoryResource & dst,
    const std::unique_ptr<nvinfer1::IExecutionContext> & exec_context,
    cudaStream_t stream
) noexcept {

    const auto set_error = [](const ErrorMessage & message) {
        return message;
    };

    auto input_name = exec_context->getEngine().getIOTensorName(0);
    auto output_name = exec_context->getEngine().getIOTensorName(1);

    if (!exec_context->setTensorAddress(input_name, src.d_data.data)) {
        return set_error("set input tensor address failed");
    }
    if (!exec_context->setTensorAddress(output_name, dst.d_data.data)) {
        return set_error("set output tensor address failed");
    }

    // flush deferred internal state update
    // https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-821/developer-guide/index.html#cuda-graphs
    {
        if (!exec_context->enqueueV3(stream)) {
            return set_error("warmup enqueue failed");
        }
        checkError(cudaStreamSynchronize(stream));
    }

    checkError(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    {
        if (!exec_context->enqueueV3(stream)) {
            return set_error("capture enqueue error");
        }
    }
    cudaGraph_t graph;
    checkError(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t graphexec;
    checkError(cudaGraphInstantiate(&graphexec, graph, nullptr, nullptr, 0));
    checkError(cudaGraphDestroy(graph));

    return graphexec;
}

static inline
size_t getSize(
    const nvinfer1::Dims & dim
) noexcept {

    size_t ret = 1;
    for (int i = 0; i < dim.nbDims; ++i) {
        ret *= dim.d[i];
    }
    return ret;
}

static inline
int getBytesPerSample(nvinfer1::DataType type) noexcept {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kBOOL:
            return 1;
        case nvinfer1::DataType::kUINT8:
            return 1;
        case nvinfer1::DataType::kFP8:
            return 1;
        case nvinfer1::DataType::kBF16:
            return 2;
        case nvinfer1::DataType::kINT64:
            return 8;
        default:
            return 0;
    }
}

static inline
std::variant<ErrorMessage, InferenceInstance> getInstance(
    const std::unique_ptr<nvinfer1::ICudaEngine> & engine,
    const std::optional<int> & profile_index,
    const TileSize & tile_size,
    bool use_cuda_graph,
    bool is_dynamic
) noexcept {

    const auto set_error = [](const ErrorMessage & error_message) {
        return error_message;
    };

    StreamResource stream {};
    checkError(cudaStreamCreateWithFlags(&stream.data, cudaStreamNonBlocking));

    StreamResource h2d_stream {};
    checkError(cudaStreamCreateWithFlags(&h2d_stream.data, cudaStreamNonBlocking));

    StreamResource d2h_stream {};
    checkError(cudaStreamCreateWithFlags(&d2h_stream.data, cudaStreamNonBlocking));

    std::array<EventResource, kNumBuffers> h2d_done {};
    std::array<EventResource, kNumBuffers> compute_done {};
    std::array<EventResource, kNumBuffers> d2h_done {};
    for (size_t b = 0; b < kNumBuffers; ++b) {
        checkError(cudaEventCreateWithFlags(&h2d_done[b].data, cudaEventDisableTiming));
        checkError(cudaEventCreateWithFlags(&compute_done[b].data, cudaEventDisableTiming));
        checkError(cudaEventCreateWithFlags(&d2h_done[b].data, cudaEventDisableTiming));
    }

    auto exec_context = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine->createExecutionContext(
            is_dynamic ?
            nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED :
            nvinfer1::ExecutionContextAllocationStrategy::kON_PROFILE_CHANGE
        )
    );

    auto input_name = engine->getIOTensorName(0);
    auto output_name = engine->getIOTensorName(1);

    if (!exec_context->allInputDimensionsSpecified()) {
        if (!profile_index.has_value()) {
            return set_error("no valid optimization profile found");
        }
        exec_context->setOptimizationProfileAsync(profile_index.value(), stream);
        checkError(cudaStreamSynchronize(stream));

        nvinfer1::Dims dims = exec_context->getTensorShape(input_name);

        dims.d[0] = 1;

        if (std::holds_alternative<RequestedTileSize>(tile_size)) {
            dims.d[2] = std::get<RequestedTileSize>(tile_size).tile_h;
            dims.d[3] = std::get<RequestedTileSize>(tile_size).tile_w;
        } else {
            dims.d[2] = std::get<VideoSize>(tile_size).height;
            dims.d[3] = std::get<VideoSize>(tile_size).width;
        }
        exec_context->setInputShape(input_name, dims);
    } else if (std::holds_alternative<RequestedTileSize>(tile_size)) {
        nvinfer1::Dims dims = exec_context->getTensorShape(input_name);

        if (std::holds_alternative<RequestedTileSize>(tile_size)) {
            if (dims.d[2] != std::get<RequestedTileSize>(tile_size).tile_h ||
                dims.d[3] != std::get<RequestedTileSize>(tile_size).tile_w
            ) {
                return set_error("requested tile size not applicable");
            }
        } else {
            if (dims.d[2] != std::get<VideoSize>(tile_size).height ||
                dims.d[3] != std::get<VideoSize>(tile_size).width
            ) {
                return set_error("not supported video dimensions");
            }
        }
    }

    std::array<MemoryResource, kNumBuffers> src {};
    {
        auto dim = exec_context->getTensorShape(input_name);
        auto type = engine->getTensorDataType(input_name);

        auto size = getSize(dim) * getBytesPerSample(type);

        for (size_t b = 0; b < kNumBuffers; ++b) {
            Resource<uint8_t *, cudaFree> d_data {};
            checkError(cudaMalloc(&d_data.data, size));

            Resource<uint8_t *, cudaFreeHost> h_data {};
            checkError(cudaMallocHost(&h_data.data, size, cudaHostAllocWriteCombined));

            src[b] = MemoryResource{
                .h_data = std::move(h_data),
                .d_data = std::move(d_data),
                .size=size
            };
        }
    }

    std::array<MemoryResource, kNumBuffers> dst {};
    {
        auto dim = exec_context->getTensorShape(output_name);
        auto type = engine->getTensorDataType(output_name);

        auto size = getSize(dim) * getBytesPerSample(type);

        for (size_t b = 0; b < kNumBuffers; ++b) {
            Resource<uint8_t *, cudaFree> d_data {};
            checkError(cudaMalloc(&d_data.data, size));

            Resource<uint8_t *, cudaFreeHost> h_data {};
            checkError(cudaMallocHost(&h_data.data, size));

            dst[b] = MemoryResource{
                .h_data = std::move(h_data),
                .d_data = std::move(d_data),
                .size=size
            };
        }
    }

    Resource<uint8_t *, cudaFree> d_context_allocation {};

    if (is_dynamic) {
        size_t buffer_size { exec_context->updateDeviceMemorySizeForShapes() };
        if (buffer_size == 0) {
            return set_error("failed to get internal activation buffer size");
        }

        checkError(cudaMalloc(&d_context_allocation.data, buffer_size));
        exec_context->setDeviceMemoryV2(d_context_allocation.data, static_cast<int64_t>(buffer_size));
    }

    std::array<GraphExecResource, kNumBuffers> graphexec {};
    if (use_cuda_graph) {
        for (size_t b = 0; b < kNumBuffers; ++b) {
            auto result = getGraphExec(
                src[b], dst[b],
                exec_context, stream
            );
            if (std::holds_alternative<GraphExecResource>(result)) {
                graphexec[b] = std::move(std::get<GraphExecResource>(result));
            } else {
                return set_error(std::get<ErrorMessage>(result));
            }
        }
    }

    return InferenceInstance{
        .src = std::move(src),
        .dst = std::move(dst),
        .stream = std::move(stream),
        .h2d_stream = std::move(h2d_stream),
        .d2h_stream = std::move(d2h_stream),
        .h2d_done = std::move(h2d_done),
        .compute_done = std::move(compute_done),
        .d2h_done = std::move(d2h_done),
        .exec_context = std::move(exec_context),
        .graphexec = std::move(graphexec),
        .d_context_allocation = std::move(d_context_allocation)
    };
}

static inline
std::optional<ErrorMessage> checkEngine(
    const std::unique_ptr<nvinfer1::ICudaEngine> & engine,
    bool flexible_output
) noexcept {

    int num_bindings = engine->getNbIOTensors();

    if (num_bindings != 2) {
        return "network binding count must be 2, got " + std::to_string(num_bindings);
    }

    auto input_name = engine->getIOTensorName(0);
    auto output_name = engine->getIOTensorName(1);

    if (engine->getTensorIOMode(input_name) != nvinfer1::TensorIOMode::kINPUT) {
        return "the first binding should be an input binding";
    }

    const nvinfer1::Dims & input_dims = engine->getTensorShape(input_name);

    if (input_dims.nbDims != 4) {
        return "expects network with 4-D input";
    }
    if (input_dims.d[0] != 1 && input_dims.d[0] != -1) {
        return "batch size of network input must be 1";
    }

    if (engine->getTensorIOMode(output_name) != nvinfer1::TensorIOMode::kOUTPUT) {
        return "the second binding should be an output binding";
    }

    const nvinfer1::Dims & output_dims = engine->getTensorShape(output_name);

    if (output_dims.nbDims != 4) {
        return "expects network with 4-D output";
    }
    if (output_dims.d[0] != 1 && output_dims.d[0] != -1) {
        return "batch size of network output must be 1";
    }

    auto out_channels = output_dims.d[1];
    if (out_channels != 1 && out_channels != 3 && !flexible_output) {
        return "output dimensions must be 1 or 3, or enable \"flexible_output\"";
    }

    auto in_height = input_dims.d[2];
    auto in_width = input_dims.d[3];
    auto out_height = output_dims.d[2];
    auto out_width = output_dims.d[3];
    if (out_height % in_height != 0 || out_width % in_width != 0) {
        return "output dimensions must be divisible by input dimensions";
    }

    for (const auto & name : { input_name, output_name }) {
        if (engine->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE) {
            return "network binding " + std::string{ name } + " should reside on device";
        }

        if (engine->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR) {
            return "expects network IO with layout NCHW (row major linear)";
        }
    }

    return {};
}

static inline
std::variant<ErrorMessage, std::unique_ptr<nvinfer1::ICudaEngine>> initEngine(
    const char * engine_data, size_t engine_nbytes,
    const std::unique_ptr<nvinfer1::IRuntime> & runtime,
    bool flexible_output
) noexcept {

    const auto set_error = [](const ErrorMessage & error_message) {
        return error_message;
    };

    std::unique_ptr<nvinfer1::ICudaEngine> engine {
        runtime->deserializeCudaEngine(engine_data, engine_nbytes)
    };

    if (!engine) {
        return set_error("engine deserialization failed");
    }

    if (auto err = checkEngine(engine, flexible_output); err.has_value()) {
        return set_error(err.value());
    }

    return engine;
}

// 0: integer, 1: float
static inline
int getSampleType(nvinfer1::DataType type) noexcept {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
        case nvinfer1::DataType::kHALF:
        case nvinfer1::DataType::kFP8:
        case nvinfer1::DataType::kBF16:
            return 1;
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kINT32:
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kUINT8:
        case nvinfer1::DataType::kINT64:
            return 0;
        default:
            return -1;
    }
}

#endif // VSTRT_TRT_UTILS_H_
