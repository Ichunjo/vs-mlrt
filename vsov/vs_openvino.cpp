#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <VapourSynth4.h>
#include <VSHelper4.h>
#include <VSConstants4.h>

#include <onnx/common/version.h>
#include <onnx/onnx_pb.h>

#include <openvino/openvino.hpp>
#include <openvino/pass/constant_folding.hpp>

#ifdef ENABLE_VISUALIZATION
#include <openvino/pass/visualize_tree.hpp>
#endif // ENABLE_VISUALIZATION

#include "../common/convert_float_to_float16.h"
#include "../common/onnx_utils.h"

#ifndef PLUGIN_VERSION_MAJOR
#define PLUGIN_VERSION_MAJOR 1
#endif
#ifndef PLUGIN_VERSION_MINOR
#define PLUGIN_VERSION_MINOR 0
#endif
#ifndef PLUGIN_VERSION_STRING
#define PLUGIN_VERSION_STRING "unknown"
#endif


using namespace std::string_literals;

#define PLUGIN_ID "io.github.amusementclub.vs_openvino"


static std::array<int, 4> getShape(
    const ov::CompiledModel & network,
    bool input
) {

    ov::Shape dims;

    if (input) {
        dims = network.input().get_shape();
    } else {
        dims = network.output().get_shape();
    }

    std::array<int, 4> ret;
    for (unsigned i = 0; i < std::size(ret); ++i) {
        ret[i] = static_cast<int>(dims[i]);
    }

    return ret;
}


static int numPlanes(
    const std::vector<const VSVideoInfo *> & vis
) {

    int num_planes = 0;

    for (const auto & vi : vis) {
        num_planes += vi->format.numPlanes;
    }

    return num_planes;
}


[[nodiscard]]
static std::optional<std::string> checkNodes(
    const std::vector<const VSVideoInfo *> & vis
) {

    for (const auto & vi : vis) {
        if (vi->format.sampleType != stFloat || vi->format.bitsPerSample != 32) {
            return "expects clip with type fp32";
        }

        if (vi->width != vis[0]->width || vi->height != vis[0]->height) {
            return "dimensions of clips mismatch";
        }

        if (vi->numFrames != vis[0]->numFrames) {
            return "number of frames mismatch";
        }

        if (vi->format.subSamplingH != 0 || vi->format.subSamplingW != 0) {
            return "clip must not be sub-sampled";
        }
    }

    return {};
}


[[nodiscard]]
static std::optional<std::string> checkIOInfo(
    const ov::Output<ov::Node> & info,
    bool is_output,
    bool flexible_output
) {

    if (info.get_element_type() != ov::element::f32) {
        return "expects network IO with type fp32";
    }
    // if (ov::layout::get_layout(info) != ov::Layout("NCHW")) {
    //     return "expects network IO with layout NCHW";
    // }
    const auto & dims = info.get_shape();
    if (dims.size() != 4) {
        return "expects network with 4-D IO";
    }

    if (dims[0] != 1) {
        return "batch size of network must be 1";
    }

    if (is_output) {
        auto out_channels = dims[1];
        if (out_channels != 1 && out_channels != 3 && !flexible_output) {
            return "output dimensions must be 1 or 3, or enable \"flexible_output\"";
        }
    }

    return {};
}


[[nodiscard]]
static std::optional<std::string> checkNetwork(
    const std::shared_ptr<ov::Model> & network,
    bool flexible_output
) {

    if (auto num_inputs = std::size(network->inputs()); num_inputs != 1) {
        return "network input count must be 1, got " + std::to_string(num_inputs);
    }

    const auto & input_info = network->input();
    if (auto err = checkIOInfo(input_info, false, flexible_output); err.has_value()) {
        return err.value();
    }

    if (auto num_outputs = std::size(network->outputs()); num_outputs != 1) {
        return "network output count must be 1, got " + std::to_string(num_outputs);
    }

    const auto & output_info = network->output();
    if (auto err = checkIOInfo(output_info, true, flexible_output); err.has_value()) {
        return err.value();
    }

    return {};
}


[[nodiscard]]
static std::optional<std::string> checkNodesAndNetwork(
    const ov::CompiledModel & network,
    const std::vector<const VSVideoInfo *> & vis
) {

    const auto & network_in_dims = (
        network.input().get_tensor().get_shape()
    );

    int network_in_channels = static_cast<int>(network_in_dims[1]);
    int num_planes = numPlanes(vis);
    if (network_in_channels != num_planes) {
        return "expects " + std::to_string(network_in_channels) + " input planes";
    }

    auto network_in_height = static_cast<int>(network_in_dims[2]);
    auto network_in_width = static_cast<int>(network_in_dims[3]);
    auto clip_in_height = vis.front()->height;
    auto clip_in_width = vis.front()->width;
    if (network_in_height > clip_in_height || network_in_width > clip_in_width) {
        return "tile size larger than clip dimension";
    }

    return {};
}


static void setDimensions(
    std::unique_ptr<VSVideoInfo> & vi,
    const ov::CompiledModel & network,
    VSCore * core,
    const VSAPI * vsapi,
    bool flexible_output
) {

    const auto & in_dims = network.input().get_shape();
    const auto & out_dims = network.output().get_shape();

    vi->height *= out_dims[2] / in_dims[2];
    vi->width *= out_dims[3] / in_dims[3];

    if (out_dims[1] == 1 || flexible_output) {
        vsapi->queryVideoFormat(&vi->format, cfGray, stFloat, 32, 0, 0, core);
    } else if (out_dims[1] == 3) {
        vsapi->queryVideoFormat(&vi->format, cfRGB, stFloat, 32, 0, 0, core);
    }
}


static std::variant<std::string, ov::AnyMap> getConfig(
    VSFunction * config_func,
    VSCore * core,
    const VSAPI * vsapi
) {

    ov::AnyMap config;

    if (config_func == nullptr) {
        return config;
    }

    auto in_map = vsapi->createMap();
    auto out_map = vsapi->createMap();

    auto set_error = [&](const std::string & error_message) -> std::string {
        vsapi->freeMap(out_map);
        vsapi->freeMap(in_map);
        return error_message;
    };

    vsapi->callFunction(config_func, in_map, out_map);

    if (auto error_message = vsapi->mapGetError(out_map); error_message) {
        return set_error(error_message);
    }

    int num_keys { vsapi->mapNumKeys(out_map) };
    for (int index = 0; index < num_keys; index++) {
        auto key = vsapi->mapGetKey(out_map, index);
        auto num_elements { vsapi->mapNumElements(out_map, key) };
        if (num_elements != 1) {
            return set_error("each value in the \"config\" dict must have exactly one element");
        }
        auto type = vsapi->mapGetType(out_map, key);
        if (type == ptData) {
            config[key] = vsapi->mapGetData(out_map, key, 0, nullptr);
        } else if (type == ptInt) {
            config[key] = std::to_string(vsapi->mapGetInt(out_map, key, 0, nullptr));
        } else if (type == ptFloat) {
            config[key] = std::to_string(vsapi->mapGetFloat(out_map, key, 0, nullptr));
        } else {
            return set_error("unknown type of key \""s + key + "\": (" + std::to_string(type) + ")");
        }
    }

    vsapi->freeMap(out_map);
    vsapi->freeMap(in_map);

    return config;
}


struct OVData {
    std::vector<VSNode *> nodes;
    std::unique_ptr<VSVideoInfo> out_vi;

    int overlap_w, overlap_h;

    ov::Core core;
    ov::CompiledModel executable_network;
    std::unordered_map<std::thread::id, ov::InferRequest> infer_requests;
    std::shared_mutex infer_requests_lock;

    std::string flexible_output_prop;
};


static const VSFrame *VS_CC vsOvGetFrame(
    int n,
    int activationReason,
    void *instanceData,
    void **frameData,
    VSFrameContext *frameCtx,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    auto *d = static_cast<OVData *>(instanceData);

    if (activationReason == arInitial) {
        for (const auto & node : d->nodes) {
            vsapi->requestFrameFilter(n, node, frameCtx);
        }
    } else if (activationReason == arAllFramesReady) {
        std::vector<const VSVideoInfo *> in_vis;
        in_vis.reserve(std::size(d->nodes));
        for (const auto & node : d->nodes) {
            in_vis.emplace_back(vsapi->getVideoInfo(node));
        }

        std::vector<const VSFrame *> src_frames;
        src_frames.reserve(std::size(d->nodes));
        for (const auto & node : d->nodes) {
            src_frames.emplace_back(vsapi->getFrameFilter(n, node, frameCtx));
        }

        auto src_stride = vsapi->getStride(src_frames.front(), 0);
        auto src_width = vsapi->getFrameWidth(src_frames.front(), 0);
        auto src_height = vsapi->getFrameHeight(src_frames.front(), 0);
        auto src_bytes = vsapi->getVideoFrameFormat(src_frames.front())->bytesPerSample;
        auto src_tile_shape = getShape(d->executable_network, true);
        auto src_tile_h = src_tile_shape[2];
        auto src_tile_w = src_tile_shape[3];
        auto src_tile_w_bytes = src_tile_w * src_bytes;
        auto src_tile_bytes = src_tile_h * src_tile_w_bytes;

        std::vector<const uint8_t *> src_ptrs;
        src_ptrs.reserve(src_tile_shape[1]);
        for (unsigned i = 0; i < std::size(d->nodes); ++i) {
            for (int j = 0; j < in_vis[i]->format.numPlanes; ++j) {
                src_ptrs.emplace_back(vsapi->getReadPtr(src_frames[i], j));
            }
        }

        auto step_w = src_tile_w - 2 * d->overlap_w;
        auto step_h = src_tile_h - 2 * d->overlap_h;

        VSFrame * const dst_frame = vsapi->newVideoFrame(
            &d->out_vi->format, d->out_vi->width, d->out_vi->height,
            src_frames.front(), core
        );

        std::vector<VSFrame *> dst_frames;

        auto dst_stride = vsapi->getStride(dst_frame, 0);
        auto dst_bytes = vsapi->getVideoFrameFormat(dst_frame)->bytesPerSample;
        auto dst_tile_shape = getShape(d->executable_network, false);
        auto dst_tile_h = dst_tile_shape[2];
        auto dst_tile_w = dst_tile_shape[3];
        auto dst_tile_w_bytes = dst_tile_w * dst_bytes;
        auto dst_tile_bytes = dst_tile_h * dst_tile_w_bytes;
        auto dst_planes = dst_tile_shape[1];

        std::vector<uint8_t *> dst_ptrs;
        if (d->flexible_output_prop.empty()) {
            for (int i = 0; i < dst_planes; ++i) {
                dst_ptrs.emplace_back(vsapi->getWritePtr(dst_frame, i));
            }
        } else {
            for (int i = 0; i < dst_planes; ++i) {
                auto frame { vsapi->newVideoFrame(
                    &d->out_vi->format, d->out_vi->width, d->out_vi->height,
                    src_frames[0], core
                )};
                dst_frames.emplace_back(frame);
                dst_ptrs.emplace_back(vsapi->getWritePtr(frame, 0));
            }
        }

        auto h_scale = dst_tile_h / src_tile_h;
        auto w_scale = dst_tile_w / src_tile_w;

        const auto set_error = [&](const std::string & error_message) {
            vsapi->setFilterError(
                (__func__ + ": "s + error_message).c_str(),
                frameCtx
            );

            vsapi->freeFrame(dst_frame);

            for (const auto & frame : dst_frames) {
                vsapi->freeFrame(frame);
            }

            for (const auto & frame : src_frames) {
                vsapi->freeFrame(frame);
            }

            return nullptr;
        };

        auto thread_id = std::this_thread::get_id();
        bool initialized = true;
        ov::InferRequest * infer_request;

        d->infer_requests_lock.lock_shared();
        try {
            infer_request = &d->infer_requests.at(thread_id);
        } catch (const std::out_of_range &) {
            initialized = false;
        }
        d->infer_requests_lock.unlock_shared();

        if (!initialized) {
            std::lock_guard _ { d->infer_requests_lock };
            try {
                d->infer_requests.emplace(thread_id, d->executable_network.create_infer_request());
            } catch (const ov::Exception & e) {
                return set_error("[OV exception] Create inference request: "s + e.what());
            } catch (const std::exception& e) {
                return set_error("[Standard exception] Create inference request: "s + e.what());
            }
            infer_request = &d->infer_requests[thread_id];
        }

        int y = 0;
        while (true) {
            int y_crop_start = (y == 0) ? 0 : d->overlap_h;
            int y_crop_end = (y == src_height - src_tile_h) ? 0 : d->overlap_h;

            int x = 0;
            while (true) {
                int x_crop_start = (x == 0) ? 0 : d->overlap_w;
                int x_crop_end = (x == src_width - src_tile_w) ? 0 : d->overlap_w;

                {
                    auto input_buffer = (uint8_t *) infer_request->get_input_tensor().data<float>();

                    for (const auto & _src_ptr : src_ptrs) {
                        const uint8_t * src_ptr { _src_ptr +
                            y * src_stride + x * src_bytes
                        };

                        vsh::bitblt(
                            input_buffer, src_tile_w_bytes,
                            src_ptr, src_stride,
                            src_tile_w_bytes, src_tile_h
                        );

                        input_buffer += src_tile_bytes;
                    }
                }

                try {
                    infer_request->infer();
                } catch (const ov::Exception & e) {
                    return set_error("[OV exception] Create inference request: "s + e.what());
                } catch (const std::exception& e) {
                    return set_error("[Standard exception] Create inference request: "s + e.what());
                }

                {
                    auto output_buffer = (const uint8_t *) infer_request->get_output_tensor().data<float>();

                    for (int plane = 0; plane < dst_planes; ++plane) {
                        uint8_t * dst_ptr = (dst_ptrs[plane] +
                            h_scale * y * dst_stride + w_scale * x * dst_bytes
                        );

                        vsh::bitblt(
                            dst_ptr + (y_crop_start * dst_stride + x_crop_start * dst_bytes),
                            dst_stride,
                            output_buffer + (y_crop_start * dst_tile_w_bytes + x_crop_start * dst_bytes),
                            dst_tile_w_bytes,
                            dst_tile_w_bytes - (x_crop_start + x_crop_end) * dst_bytes,
                            dst_tile_h - (y_crop_start + y_crop_end)
                        );

                        output_buffer += dst_tile_bytes;
                    }
                }

                if (x + src_tile_w == src_width) {
                    break;
                }

                x = std::min(x + step_w, src_width - src_tile_w);
            }

            if (y + src_tile_h == src_height) {
                break;
            }

            y = std::min(y + step_h, src_height - src_tile_h);
        }

        for (const auto & frame : src_frames) {
            vsapi->freeFrame(frame);
        }

        if (!d->flexible_output_prop.empty()) {
            auto prop = vsapi->getFramePropertiesRW(dst_frame);

            for (int i = 0; i < dst_planes; i++) {
                auto key { d->flexible_output_prop + std::to_string(i) };
                vsapi->mapSetFrame(prop, key.c_str(), dst_frames[i], maReplace);
                vsapi->freeFrame(dst_frames[i]);
            }
        }

        return dst_frame;
    }

    return nullptr;
}


static void VS_CC vsOvFree(
    void *instanceData,
    VSCore *core,
    const VSAPI *vsapi
) noexcept {

    OVData * d = static_cast<OVData *>(instanceData);

    for (const auto & node : d->nodes) {
        vsapi->freeNode(node);
    }

    delete d;
}


static void VS_CC vsOvCreate(
    const VSMap *in,
    VSMap *out,
    void *userData,
    VSCore *core,
    const VSAPI *vsapi
) {

    std::unique_ptr<OVData> d = nullptr;

    try {
        d = std::make_unique<OVData>();
    } catch (const ov::Exception& e) {
        vsapi->mapSetError(out, ("[OV exception] Initialize inference engine: "s + e.what()).c_str());
        return ;
    } catch (const std::exception& e) {
        vsapi->mapSetError(out, ("[Standard exception] Initialize inference engine: "s + e.what()).c_str());
        return ;
    }

    int num_nodes = vsapi->mapNumElements(in, "clips");
    d->nodes.reserve(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        d->nodes.emplace_back(vsapi->mapGetNode(in, "clips", i, nullptr));
    }

    const auto set_error = [&](const std::string & error_message) {
        vsapi->mapSetError(out, (__func__ + ": "s + error_message).c_str());
        for (const auto & node : d->nodes) {
            vsapi->freeNode(node);
        }
    };

    std::vector<const VSVideoInfo *> in_vis;
    in_vis.reserve(std::size(d->nodes));
    for (const auto & node : d->nodes) {
        in_vis.emplace_back(vsapi->getVideoInfo(node));
    }

    if (auto err = checkNodes(in_vis); err.has_value()) {
        return set_error(err.value());
    }

    d->out_vi = std::make_unique<VSVideoInfo>(*in_vis.front()); // mutable

    int error;

    const char * device = vsapi->mapGetData(in, "device", 0, &error);
    if (error) {
        device = "CPU";
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
    if (!error1) { // manual specification triggered
        if (error2) {
            tile_h = tile_w;
        }
    } else {
        if (d->overlap_w != 0 || d->overlap_h != 0) {
            return set_error("\"tilesize\" must be specified");
        }

        // set tile size to video dimensions
        tile_w = in_vis.front()->width;
        tile_h = in_vis.front()->height;
    }
    if (tile_w - 2 * d->overlap_w <= 0 || tile_h - 2 * d->overlap_h <= 0) {
        return set_error("\"overlap\" too large");
    }

    bool fp16 = !!vsapi->mapGetInt(in, "fp16", 0, &error);
    if (error) {
        fp16 = false;
    }

    bool path_is_serialization = !!vsapi->mapGetInt(in, "path_is_serialization", 0, &error);
    if (error) {
        path_is_serialization = false;
    }

    std::string_view path_view;
    std::string path;
    if (path_is_serialization) {
        path_view = {
            vsapi->mapGetData(in, "network_path", 0, nullptr),
            static_cast<size_t>(vsapi->mapGetDataSize(in, "network_path", 0, nullptr))
        };
    } else {
        path = vsapi->mapGetData(in, "network_path", 0, nullptr);
        path_view = path;
    }

    auto flexible_output_prop = vsapi->mapGetData(in, "flexible_output_prop", 0, &error);
    if (!error) {
        d->flexible_output_prop = flexible_output_prop;
    }

    auto result = loadONNX(path_view, tile_w, tile_h, path_is_serialization);
    if (std::holds_alternative<std::string>(result)) {
        return set_error(std::get<std::string>(result));
    }

    auto onnx_model = std::move(std::get<ONNX_NAMESPACE::ModelProto>(result));

    if (fp16) {
        std::unordered_set<std::string> fp16_blacklist_ops;
        int num = vsapi->mapNumElements(in, "fp16_blacklist_ops");
        if (num == -1) {
            fp16_blacklist_ops = {
                "ArrayFeatureExtractor", "Binarizer", "CastMap", "CategoryMapper",
                "DictVectorizer", "FeatureVectorizer", "Imputer", "LabelEncoder",
                "LinearClassifier", "LinearRegressor", "Normalizer", "OneHotEncoder",
                "SVMClassifier", "SVMRegressor", "Scaler", "TreeEnsembleClassifier",
                "TreeEnsembleRegressor", "ZipMap", "NonMaxSuppression", "TopK",
                "RoiAlign", "Range", "CumSum", "Min", "Max"
            };
        } else {
            for (int i = 0; i < num; i++) {
                fp16_blacklist_ops.emplace(vsapi->mapGetData(in, "fp16_blacklist_ops", i, nullptr));
            }
        }
        convert_float_to_float16(onnx_model, false, fp16_blacklist_ops);
    }

    std::string onnx_data = onnx_model.SerializeAsString();
    if (std::size(onnx_data) == 0) {
        return set_error("proto serialization failed");
    }

    {
        std::shared_ptr<ov::Model> network;
        try {
            network = d->core.read_model(onnx_data, ov::Tensor());
        } catch (const ov::Exception& e) {
            return set_error("[OV exception] ReadNetwork(): "s + e.what());
        } catch (const std::exception& e) {
            return set_error("[Standard exception] ReadNetwork(): "s + e.what());
        }

        if (auto err = checkNetwork(network, !d->flexible_output_prop.empty()); err.has_value()) {
            return set_error(err.value());
        }

        try {
            ov::pass::ConstantFolding().run_on_model(network);
        } catch (const ov::Exception & e) {
            return set_error(e.what());
        }

#ifdef ENABLE_VISUALIZATION
        const char * dot_path = vsapi->mapGetData(in, "dot_path", 0, &error);
        if (!error) {
            try {
                ov::pass::VisualizeTree(dot_path, nullptr, true).run_on_model(network);
            } catch (const ov::Exception & e) {
                return set_error(e.what());
            }
        }
#endif // ENABLE_VISUALIZATION

        auto config_func = vsapi->mapGetFunction(in, "config", 0, &error);
        auto config_ret = getConfig(config_func, core, vsapi);
        vsapi->freeFunction(config_func);
        if (std::holds_alternative<std::string>(config_ret)) {
            return set_error(std::get<std::string>(config_ret));
        }
        auto & config = std::get<ov::AnyMap>(config_ret);

        try {
            d->executable_network = d->core.compile_model(network, device, config);
        } catch (const ov::Exception & e) {
            return set_error(e.what());
        }

        if (auto err = checkNodesAndNetwork(d->executable_network, in_vis); err.has_value()) {
            return set_error(err.value());
        }

        setDimensions(d->out_vi, d->executable_network, core, vsapi, !d->flexible_output_prop.empty());

        VSCoreInfo core_info;
        vsapi->getCoreInfo(core, &core_info);
        d->infer_requests.reserve(core_info.numThreads);
    }

    if (!d->flexible_output_prop.empty()) {
        auto num_planes = d->executable_network.output(0).get_shape()[1];
        vsapi->mapSetInt(out, "num_planes", static_cast<int>(num_planes), maReplace);
    }

    std::vector<VSFilterDependency> deps;
    deps.reserve(d->nodes.size());
    for (auto *node : d->nodes) {
        deps.push_back({node, rpGeneral});
    }

    auto *out_vi = d->out_vi.get();
    auto *instance_data = d.release();

    vsapi->createVideoFilter(
        out,
        "Model",
        out_vi,
        vsOvGetFrame,
        vsOvFree,
        fmParallel,
        deps.data(),
        static_cast<int>(deps.size()),
        instance_data,
        core
    );
}


VS_EXTERNAL_API(void) VapourSynthPluginInit2(
    VSPlugin *plugin,
    const VSPLUGINAPI *vspapi
) {
    vspapi->configPlugin(
        PLUGIN_ID,
        "ov",
        "OpenVINO ML Filter Runtime",
        VS_MAKE_VERSION(PLUGIN_VERSION_MAJOR, PLUGIN_VERSION_MINOR),
        VAPOURSYNTH_API_VERSION,
        0,
        plugin
    );

    vspapi->registerFunction(
        "Model",
        "clips:vnode[];"
        "network_path:data;"
        "overlap:int[]:opt;"
        "tilesize:int[]:opt;"
        "device:data:opt;" // "CPU": CPU
        "fp16:int:opt;"
        "config:func:opt;"
        "path_is_serialization:int:opt;"
        "fp16_blacklist_ops:data[]:opt;"
#ifdef ENABLE_VISUALIZATION
        "dot_path:data:opt;"
#endif
        "flexible_output_prop:data:opt;",
        "any",
        vsOvCreate,
        nullptr,
        plugin
    );

    auto getVersion = [](const VSMap *, VSMap * out, void *, VSCore * core, const VSAPI *vsapi) {
        vsapi->mapSetData(out, "version", PLUGIN_VERSION_STRING, -1, dtUtf8, maReplace);

        std::ostringstream ostream;
        ostream << OPENVINO_VERSION_MAJOR << '.' << OPENVINO_VERSION_MINOR << '.' << OPENVINO_VERSION_PATCH;
        vsapi->mapSetData(out, "openvino_version_build", ostream.str().c_str(), -1, dtUtf8, maReplace);

        vsapi->mapSetData(out, "openvino_version", ov::get_openvino_version().buildNumber, -1, dtUtf8, maReplace);

        vsapi->mapSetData(
            out, "onnx_version",
            ONNX_NAMESPACE::LAST_RELEASE_VERSION, -1, dtUtf8, maReplace
        );

#ifdef ENABLE_VISUALIZATION
        vsapi->mapSetInt(out, "enable_visualization", 1, maReplace);
#endif // ENABLE_VISUALIZATION

        auto plugin = vsapi->getPluginByID(PLUGIN_ID, core);
        if (plugin) {
            vsapi->mapSetData(out, "path", vsapi->getPluginPath(plugin), -1, dtUtf8, maReplace);
        }
    };
    vspapi->registerFunction("Version", "", "any", getVersion, nullptr, plugin);

    auto availableDevices = [](const VSMap *, VSMap * out, void *, VSCore *, const VSAPI *vsapi) {
        try {
            auto core = ov::Core();
            auto devices = core.get_available_devices();
            for (const auto & device : devices) {
                vsapi->mapSetData(out, "devices", device.c_str(), -1, dtUtf8, maAppend);
            }
        } catch (const ov::Exception& e) {
            vsapi->mapSetError(out, ("[OV exception] Initialize inference engine: "s + e.what()).c_str());
        } catch (const std::exception& e) {
            vsapi->mapSetError(out, ("[Standard exception] Initialize inference engine: "s + e.what()).c_str());
        }
    };
    vspapi->registerFunction("AvailableDevices", "", "any", availableDevices, nullptr, plugin);
}
