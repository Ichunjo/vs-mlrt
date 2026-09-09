#include <NvInferRuntime.h>
#include <VSConstants4.h>
#include <VSHelper4.h>
#include <VapourSynth4.h>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <ios>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#ifdef USE_NVINFER_PLUGIN
#include <NvInferPlugin.h>
#endif

#ifndef PLUGIN_VERSION_MAJOR
#define PLUGIN_VERSION_MAJOR 1
#endif
#ifndef PLUGIN_VERSION_MINOR
#define PLUGIN_VERSION_MINOR 0
#endif
#ifndef PLUGIN_VERSION_STRING
#define PLUGIN_VERSION_STRING "unknown"
#endif
#include "inference_helper.h"
#include "trt_utils.h"
#include "utils.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

static std::wstring translateName(const char* name) {
    auto size = MultiByteToWideChar(CP_UTF8, 0, name, -1, nullptr, 0);
    std::wstring ret(static_cast<size_t>(size), {});
    MultiByteToWideChar(CP_UTF8, 0, name, -1, ret.data(), size);
    return ret;
}
#else
#define translateName(n) (n)
#endif

using namespace std::string_literals;

#if defined(TRT_MAJOR_RTX)
#define PLUGIN_ID "io.github.amusementclub.vs_tensorrt_rtx"
#else
#define PLUGIN_ID "io.github.amusementclub.vs_tensorrt"
#endif

struct TicketSemaphore {
    std::atomic<intptr_t> ticket{};
    std::atomic<intptr_t> current{};

    void init(intptr_t num) noexcept { current.store(num, std::memory_order::seq_cst); }

    void acquire() noexcept {
        intptr_t tk{ticket.fetch_add(1, std::memory_order::acquire)};
        while (true) {
            intptr_t curr{current.load(std::memory_order::acquire)};
            if (tk < curr) {
                return;
            }
            current.wait(curr, std::memory_order::relaxed);
        }
    }

    void release() noexcept {
        current.fetch_add(1, std::memory_order::release);
        current.notify_all();
    }
};

std::unique_ptr<Logger> logger;

struct vsTrtData {
    std::vector<VSNode*> nodes;
    std::unique_ptr<VSVideoInfo> out_vi;

    int device_id;
    int num_streams;
    bool use_cuda_graph;
    int overlap_w, overlap_h;

    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::vector<std::unique_ptr<nvinfer1::ICudaEngine>> engines;

    TicketSemaphore semaphore;
    std::vector<int> tickets;
    std::mutex instances_lock;
    std::vector<InferenceInstance> instances;

    std::string flexible_output_prop;

    [[nodiscard]]
    int acquire() noexcept {
        semaphore.acquire();
        int ticket;
        {
            std::lock_guard<std::mutex> lock{instances_lock};
            ticket = tickets.back();
            tickets.pop_back();
        }
        return ticket;
    }

    void release(int ticket) noexcept {
        {
            std::lock_guard<std::mutex> lock{instances_lock};
            tickets.push_back(ticket);
        }
        semaphore.release();
    }
};

static const VSFrame* VS_CC vsTrtGetFrame(
    int n,
    int activationReason,
    void* instanceData,
    void** frameData,
    VSFrameContext* frameCtx,
    VSCore* core,
    const VSAPI* vsapi
) noexcept {

    auto d = static_cast<vsTrtData*>(instanceData);

    if (activationReason == arInitial) {
        for (const auto& node : d->nodes) {
            vsapi->requestFrameFilter(n, node, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        const std::vector<const VSVideoInfo*> in_vis{getVideoInfo(vsapi, d->nodes)};

        const std::vector<const VSFrame*> src_frames{getFrames(n, vsapi, frameCtx, d->nodes)};

        const int ticket{d->acquire()};
        InferenceInstance& instance{d->instances[ticket]};

        auto input_name = d->engines[0]->getIOTensorName(0);
        const nvinfer1::Dims src_dim{instance.exec_context->getTensorShape(input_name)};

        const int src_planes{static_cast<int>(src_dim.d[1])};
        const int src_tile_h{static_cast<int>(src_dim.d[2])};
        const int src_tile_w{static_cast<int>(src_dim.d[3])};

        std::vector<const uint8_t*> src_ptrs;
        src_ptrs.reserve(src_planes);
        for (int i = 0; i < std::ssize(d->nodes); ++i) {
            for (int j = 0; j < in_vis[i]->format.numPlanes; ++j) {
                src_ptrs.emplace_back(vsapi->getReadPtr(src_frames[i], j));
            }
        }

        VSFrame* const dst_frame{
            vsapi->newVideoFrame(&d->out_vi->format, d->out_vi->width, d->out_vi->height, src_frames[0], core)
        };

        std::vector<VSFrame*> dst_frames;

        auto output_name = d->engines[0]->getIOTensorName(1);
        const nvinfer1::Dims dst_dim{instance.exec_context->getTensorShape(output_name)};

        const int dst_planes{static_cast<int>(dst_dim.d[1])};
        const int dst_tile_h{static_cast<int>(dst_dim.d[2])};
        const int dst_tile_w{static_cast<int>(dst_dim.d[3])};

        std::vector<uint8_t*> dst_ptrs;
        dst_ptrs.reserve(dst_planes);
        if (d->flexible_output_prop.empty()) {
            for (int i = 0; i < dst_planes; ++i) {
                dst_ptrs.emplace_back(vsapi->getWritePtr(dst_frame, i));
            }
        } else {
            for (int i = 0; i < dst_planes; ++i) {
                auto frame{
                    vsapi->newVideoFrame(&d->out_vi->format, d->out_vi->width, d->out_vi->height, src_frames[0], core)
                };
                dst_frames.emplace_back(frame);
                dst_ptrs.emplace_back(vsapi->getWritePtr(frame, 0));
            }
        }

        const int h_scale = dst_tile_h / src_tile_h;
        const int w_scale = dst_tile_w / src_tile_w;

        const IOInfo info{
            .in =
                InputInfo{
                    .width = vsapi->getFrameWidth(src_frames[0], 0),
                    .height = vsapi->getFrameHeight(src_frames[0], 0),
                    .pitch = vsapi->getStride(src_frames[0], 0),
                    .bytes_per_sample = vsapi->getVideoFrameFormat(src_frames[0])->bytesPerSample,
                    .tile_w = src_tile_w,
                    .tile_h = src_tile_h
                },
            .out =
                OutputInfo{
                    .pitch = vsapi->getStride(dst_frame, 0),
                    .bytes_per_sample = vsapi->getVideoFrameFormat(dst_frame)->bytesPerSample
                },
            .w_scale = w_scale,
            .h_scale = h_scale,
            .overlap_w = d->overlap_w,
            .overlap_h = d->overlap_h
        };

        const auto inference_result = inference(instance, d->device_id, d->use_cuda_graph, info, src_ptrs, dst_ptrs);

        d->release(ticket);

        for (const auto& frame : src_frames) {
            vsapi->freeFrame(frame);
        }

        if (inference_result.has_value()) {
            vsapi->setFilterError((__func__ + ": "s + inference_result.value()).c_str(), frameCtx);

            for (const auto& frame : dst_frames) {
                vsapi->freeFrame(frame);
            }

            vsapi->freeFrame(dst_frame);

            return nullptr;
        }

        if (!d->flexible_output_prop.empty()) {
            auto prop = vsapi->getFramePropertiesRW(dst_frame);

            for (int i = 0; i < dst_planes; i++) {
                auto key{d->flexible_output_prop + std::to_string(i)};
                vsapi->mapSetFrame(prop, key.c_str(), dst_frames[i], maReplace);
                vsapi->freeFrame(dst_frames[i]);
            }
        }

        return dst_frame;
    }

    return nullptr;
}

static void VS_CC vsTrtFree(void* instanceData, VSCore* core, const VSAPI* vsapi) noexcept {

    auto d = static_cast<vsTrtData*>(instanceData);

    for (const auto& node : d->nodes) {
        vsapi->freeNode(node);
    }

    cudaSetDevice(d->device_id);

    delete d;
}

static void VS_CC vsTrtCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) noexcept {

    auto d{std::make_unique<vsTrtData>()};

    int num_nodes = vsapi->mapNumElements(in, "clips");
    d->nodes.reserve(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        d->nodes.emplace_back(vsapi->mapGetNode(in, "clips", i, nullptr));
    }

    auto set_error = [&](const std::string& error_message) {
        vsapi->mapSetError(out, (__func__ + ": "s + error_message).c_str());
        for (const auto& node : d->nodes) {
            vsapi->freeNode(node);
        }
    };

    const char* engine_path = vsapi->mapGetData(in, "engine_path", 0, nullptr);

    std::vector<const VSVideoInfo*> in_vis;
    in_vis.reserve(std::size(d->nodes));
    for (const auto& node : d->nodes) {
        in_vis.emplace_back(vsapi->getVideoInfo(node));
    }
    if (auto err = checkNodes(in_vis); err.has_value()) {
        return set_error(err.value());
    }

    int error1, error2;
    d->overlap_w = vsapi->mapGetIntSaturated(in, "overlap", 0, &error1);
    d->overlap_h = vsapi->mapGetIntSaturated(in, "overlap", 1, &error2);
    if (!error1) {
        if (error2) {
            d->overlap_h = d->overlap_w;
        }

        if (d->overlap_w < 0 || d->overlap_h < 0) {
            return set_error("\"overlap\" must be non-negative");
        }
    } else {
        d->overlap_w = 0;
        d->overlap_h = 0;
    }

    int tile_w = vsapi->mapGetIntSaturated(in, "tilesize", 0, &error1);
    int tile_h = vsapi->mapGetIntSaturated(in, "tilesize", 1, &error2);

    TileSize tile_size;
    if (!error1) { // manual specification triggered
        if (error2) {
            tile_h = tile_w;
        }

        if (tile_w - 2 * d->overlap_w <= 0 || tile_h - 2 * d->overlap_h <= 0) {
            return set_error("\"overlap\" too large");
        }

        tile_size = RequestedTileSize{.tile_w = tile_w, .tile_h = tile_h};
    } else {
        if (d->overlap_w != 0 || d->overlap_h != 0) {
            return set_error("\"tilesize\" must be specified");
        }

        int width = in_vis[0]->width;
        int height = in_vis[0]->height;

        if (width - 2 * d->overlap_w <= 0 || height - 2 * d->overlap_h <= 0) {
            return set_error("\"overlap\" too large");
        }

        tile_size = VideoSize{.width = width, .height = height};
    }

    int error;

    int device_id = vsapi->mapGetIntSaturated(in, "device_id", 0, &error);
    if (error) {
        device_id = 0;
    }

    int device_count;
    checkError(cudaGetDeviceCount(&device_count));
    if (0 <= device_id && device_id < device_count) {
        checkError(cudaSetDevice(device_id));
    } else {
        return set_error("invalid device ID (" + std::to_string(device_id) + ")");
    }
    d->device_id = device_id;

    d->use_cuda_graph = !!vsapi->mapGetInt(in, "use_cuda_graph", 0, &error);
    if (error) {
        d->use_cuda_graph = false;
    }

    d->num_streams = vsapi->mapGetIntSaturated(in, "num_streams", 0, &error);
    if (error) {
        d->num_streams = 2;
    }

    int verbosity = vsapi->mapGetIntSaturated(in, "verbosity", 0, &error);
    if (error) {
        verbosity = int(nvinfer1::ILogger::Severity::kWARNING);
    }
    if (!logger) {
        logger = std::make_unique<Logger>();
    }
    logger->set_verbosity(static_cast<nvinfer1::ILogger::Severity>(verbosity));
    logger->set_vs_api(vsapi, core);

    auto flexible_output_prop = vsapi->mapGetData(in, "flexible_output_prop", 0, &error);
    if (!error) {
        d->flexible_output_prop = flexible_output_prop;
    }

#ifdef USE_NVINFER_PLUGIN
#if !defined(TRT_MAJOR_RTX)
    if (!initLibNvInferPlugins(logger.get(), "")) {
        vsapi->logMessage(mtWarning, "vstrt: Initialize TensorRT plugins failed", core);
    }
#endif
#endif

    std::ifstream engine_stream{translateName(engine_path), std::ios::binary | std::ios::ate};

    if (!engine_stream.good()) {
        return set_error("open engine failed");
    }

    auto engine_nbytes = engine_stream.tellg();
    if (engine_nbytes == -1) {
        return set_error("open engine failed");
    }

    std::unique_ptr<char[], decltype(&free)> engine_data{(char*)malloc(static_cast<size_t>(engine_nbytes)), free};
    engine_stream.seekg(0, std::ios::beg);
    engine_stream.read(engine_data.get(), static_cast<std::streamsize>(engine_nbytes));

    d->runtime.reset(nvinfer1::createInferRuntime(*logger));
#if defined(TRT_MAJOR_RTX) && NV_TENSORRT_VERSION >= 10100
    if (static_cast<int64_t>(engine_nbytes) < d->runtime->getEngineHeaderSize()) {
        return set_error("invalid engine size: " + std::to_string(engine_nbytes));
    }
    {
        uint64_t diagnostics;
        auto engine_validity =
            d->runtime->getEngineValidity(engine_data.get(), static_cast<int64_t>(engine_nbytes), &diagnostics);
        if (engine_validity == nvinfer1::EngineValidity::kSUBOPTIMAL) {
            vsapi->logMessage(mtWarning, "suboptimal engine", core);
        } else if (engine_validity == nvinfer1::EngineValidity::kINVALID) {
            std::ostringstream diagnostics_message;
            diagnostics_message << "invalid engine: ";
            if (diagnostics & static_cast<uint64_t>(nvinfer1::EngineInvalidityDiagnostics::kVERSION_MISMATCH)) {
                diagnostics_message << "trt version mismatch, ";
            }
            if (diagnostics & static_cast<uint64_t>(nvinfer1::EngineInvalidityDiagnostics::kUNSUPPORTED_CC)) {
                diagnostics_message << "unsupported compute capability, ";
            }
            if (diagnostics & static_cast<uint64_t>(nvinfer1::EngineInvalidityDiagnostics::kOLD_CUDA_DRIVER)) {
                diagnostics_message << "cuda driver too old, ";
            }
            if (diagnostics & static_cast<uint64_t>(nvinfer1::EngineInvalidityDiagnostics::kOLD_CUDA_RUNTIME)) {
                diagnostics_message << "cuda runtime too old, ";
            }
            if (diagnostics & static_cast<uint64_t>(nvinfer1::EngineInvalidityDiagnostics::kINSUFFICIENT_GPU_MEMORY)) {
                diagnostics_message << "insufficient gpu memory, ";
            }
            if (diagnostics & static_cast<uint64_t>(nvinfer1::EngineInvalidityDiagnostics::kMALFORMED_ENGINE)) {
                diagnostics_message << "malformed engine, ";
            }
            if (diagnostics & static_cast<uint64_t>(nvinfer1::EngineInvalidityDiagnostics::kCUDA_ERROR)) {
                diagnostics_message << "cuda error, ";
            }
            return set_error(diagnostics_message.str());
        }
    }
#endif
    auto maybe_engine =
        initEngine(engine_data.get(), static_cast<size_t>(engine_nbytes), d->runtime, !d->flexible_output_prop.empty());
    if (std::holds_alternative<std::unique_ptr<nvinfer1::ICudaEngine>>(maybe_engine)) {
        d->engines.push_back(std::move(std::get<std::unique_ptr<nvinfer1::ICudaEngine>>(maybe_engine)));
    } else {
        return set_error(std::get<ErrorMessage>(maybe_engine));
    }

    auto maybe_profile_index = selectProfile(d->engines[0], tile_size);

    bool is_dynamic = false;
    {
        auto input_name = d->engines[0]->getIOTensorName(0);
        auto input_shape = d->engines[0]->getTensorShape(input_name);
        for (int32_t i = 0; i < input_shape.nbDims; i++) {
            if (input_shape.d[i] == -1) {
                is_dynamic = true;
                break;
            }
        }
    }

    d->instances.reserve(d->num_streams);
    for (int i = 0; i < d->num_streams; ++i) {
        auto maybe_instance =
            getInstance(d->engines.back(), maybe_profile_index, tile_size, d->use_cuda_graph, is_dynamic);

        if (std::holds_alternative<InferenceInstance>(maybe_instance)) {
            auto instance = std::move(std::get<InferenceInstance>(maybe_instance));
            if (auto err = checkNodesAndContext(instance.exec_context, in_vis); err.has_value()) {
                return set_error(err.value());
            }
            d->instances.emplace_back(std::move(instance));
        } else {
            return set_error(std::get<ErrorMessage>(maybe_instance));
        }
    }

    d->semaphore.init(d->num_streams);
    d->tickets.reserve(d->num_streams);
    for (int i = 0; i < d->num_streams; ++i) {
        d->tickets.push_back(i);
    }

    auto input_name = d->engines[0]->getIOTensorName(0);
    auto input_type = d->engines[0]->getTensorDataType(input_name);

    VSSampleType input_sample_type;
    {
        auto sample_type = getSampleType(input_type);
        if (sample_type == 0) {
            input_sample_type = stInteger;
        } else if (sample_type == 1) {
            input_sample_type = stFloat;
        } else {
            return set_error("unknown input sample type");
        }
    }
    auto input_bits_per_sample = getBytesPerSample(input_type) * 8;

    if (auto err = checkNodes(in_vis, input_sample_type, input_bits_per_sample); err.has_value()) {
        return set_error(err.value());
    }

    d->out_vi = std::make_unique<VSVideoInfo>(*in_vis[0]);

    auto output_name = d->engines[0]->getIOTensorName(1);
    auto output_type = d->engines[0]->getTensorDataType(output_name);

    VSSampleType output_sample_type;
    {
        auto sample_type = getSampleType(output_type);
        if (sample_type == 0) {
            output_sample_type = stInteger;
        } else if (sample_type == 1) {
            output_sample_type = stFloat;
        } else {
            return set_error("unknown output sample type");
        }
    }
    auto output_bits_per_sample = getBytesPerSample(output_type) * 8;

    setDimensions(
        d->out_vi,
        d->instances[0].exec_context,
        core,
        vsapi,
        output_sample_type,
        output_bits_per_sample,
        !d->flexible_output_prop.empty()
    );

    if (!d->flexible_output_prop.empty()) {
        const auto& exec_context = d->instances[0].exec_context;
        const nvinfer1::Dims& out_dims = exec_context->getTensorShape(output_name);
        vsapi->mapSetInt(out, "num_planes", out_dims.d[1], maReplace);
    }

    std::vector<VSFilterDependency> deps;
    deps.reserve(d->nodes.size());
    for (auto* node : d->nodes) {
        deps.push_back({node, rpGeneral});
    }

    auto* out_vi = d->out_vi.get();
    auto* instance_data = d.release();

    vsapi->createVideoFilter(
        out,
        "Model",
        out_vi,
        vsTrtGetFrame,
        vsTrtFree,
        fmParallel,
        deps.data(),
        static_cast<int>(deps.size()),
        instance_data,
        core
    );
}

VS_EXTERNAL_API(void)
VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin(
        PLUGIN_ID,
#if defined(TRT_MAJOR_RTX)
        "trt_rtx",
        "TensorRT-RTX ML Filter Runtime",
#else
        "trt",
        "TensorRT ML Filter Runtime",
#endif
        VS_MAKE_VERSION(PLUGIN_VERSION_MAJOR, PLUGIN_VERSION_MINOR),
        VAPOURSYNTH_API_VERSION,
        0,
        plugin
    );

    int ver = getInferLibVersion();
#ifdef _WIN32
    if (ver == 0) {
#if defined(TRT_MAJOR_RTX)
        std::fprintf(stderr, "vstrt_rtx: TensorRT-RTX failed to load.\n");
#else
        std::fprintf(stderr, "vstrt: TensorRT failed to load.\n");
#endif
        return;
    }
#endif // _WIN32
    if (ver != NV_TENSORRT_VERSION) {
#if defined(TRT_MAJOR_RTX)
        std::fprintf(
            stderr,
            "vstrt_rtx: TensorRT-RTX version mismatch, built with %ld but "
            "loaded with %d; continue but fingers crossed...\n",
            static_cast<long>(NV_TENSORRT_VERSION),
            ver
        );
#else
        std::fprintf(
            stderr,
            "vstrt: TensorRT version mismatch, built with %ld but loaded "
            "with %d; continue but fingers crossed...\n",
            static_cast<long>(NV_TENSORRT_VERSION),
            ver
        );
#endif
    }

    vspapi->registerFunction(
        "Model",
        "clips:vnode[];"
        "engine_path:data;"
        "overlap:int[]:opt;"
        "tilesize:int[]:opt;"
        "device_id:int:opt;"
        "use_cuda_graph:int:opt;"
        "num_streams:int:opt;"
        "verbosity:int:opt;"
        "flexible_output_prop:data:opt;",
        "any",
        vsTrtCreate,
        nullptr,
        plugin
    );

    auto getVersion = [](const VSMap*, VSMap* out, void*, VSCore* core, const VSAPI* vsapi) {
        vsapi->mapSetData(out, "version", PLUGIN_VERSION_STRING, -1, dtUtf8, maReplace);

        vsapi->mapSetData(out, "tensorrt_version", std::to_string(getInferLibVersion()).c_str(), -1, dtUtf8, maReplace);

        vsapi->mapSetData(
            out, "tensorrt_version_build", std::to_string(NV_TENSORRT_VERSION).c_str(), -1, dtUtf8, maReplace
        );

        int runtime_version;
        cudaRuntimeGetVersion(&runtime_version);
        vsapi->mapSetData(out, "cuda_runtime_version", std::to_string(runtime_version).c_str(), -1, dtUtf8, maReplace);

        vsapi->mapSetData(
            out, "cuda_runtime_version_build", std::to_string(__CUDART_API_VERSION).c_str(), -1, dtUtf8, maReplace
        );

        auto plugin = vsapi->getPluginByID(PLUGIN_ID, core);
        if (plugin) {
            vsapi->mapSetData(out, "path", vsapi->getPluginPath(plugin), -1, dtUtf8, maReplace);
        }
    };
    vspapi->registerFunction("Version", "", "any", getVersion, nullptr, plugin);

    vspapi->registerFunction("DeviceProperties", "device_id:int:opt;", "any", getDeviceProp, nullptr, plugin);
}
