#ifndef VSTRT_GPU_HELPER_H_
#define VSTRT_GPU_HELPER_H_

#include <VSVulkan4.h>
#include <algorithm>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "cuda_helper.h"
#include "cuda_utils.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

static inline std::optional<std::string> selectCUDADevice(const VSVulkanCoreInfo* info, int& device_id, bool strict) {
    auto set_error = [](const std::string& error) { return error; };
    int count{};
    checkError(cudaGetDeviceCount(&count));
    for (int i = 0; info && i < count; ++i) {
        cudaDeviceProp prop{};
        checkError(cudaGetDeviceProperties(&prop, i));
        if (!std::memcmp(prop.uuid.bytes, info->deviceUUID, VK_UUID_SIZE)) {
            device_id = i;
            return {};
        }
    }
    if (strict)
        return "the input Vulkan device has no matching CUDA device";
    if (count == 0)
        return "no CUDA device is available for TensorRT";
    device_id = 0;
    return {};
}

struct GPUMemoryAccounting {
    const VSVULKANAPI* vkapi{};
    VSGPUMemoryReservation* reservation{};
    int64_t bytes{};

    ~GPUMemoryAccounting() {
        if (reservation)
            vkapi->releaseGPUMemoryReservation(reservation);
    }

    void init(const VSAPI* vsapi, VSCore* core, int device_id) {
        vkapi = vsapi->getVulkanAPI();
        VSVulkanCoreInfo info{};
        cudaDeviceProp prop{};
        char error[512]{};
        // CPU inference still works without vulkan
        if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess ||
            vkapi->getVulkanCoreInfo(core, &info, error, sizeof(error)) ||
            std::memcmp(prop.uuid.bytes, info.deviceUUID, VK_UUID_SIZE))
            return;
        reservation = vkapi->reserveGPUMemory(core, 0, error, sizeof(error));
    }

    void reserve(size_t size) {
        if (reservation) {
            bytes += static_cast<int64_t>(size);
            vkapi->updateGPUMemoryReservation(reservation, bytes);
        }
    }
};

// NT handles remain ours after import, CUDA takes ownership of a successfully imported fd
struct GPUExportHandle {
    intptr_t handle;
    ~GPUExportHandle() {
#ifdef _WIN32
        CloseHandle(reinterpret_cast<HANDLE>(handle));
#else
        if (handle >= 0)
            close(static_cast<int>(handle));
#endif
    }
    void imported() noexcept {
#ifndef _WIN32
        handle = -1;
#endif
    }
};

struct GPUMemoryImport {
    uint64_t id{};
    const VSVULKANAPI* vkapi{};
    VSGPUMemoryReservation* reservation{};
    Resource<cudaExternalMemory_t, cudaDestroyExternalMemory> memory{};
    Resource<uint8_t*, cudaFree> base{};
    ~GPUMemoryImport() {
        base = nullptr;
        memory = nullptr;
        if (reservation)
            vkapi->releaseGPUMemoryReservation(reservation);
    }
};

struct GPUSemaphoreImport {
    VkSemaphore semaphore{};
    Resource<cudaExternalSemaphore_t, cudaDestroyExternalSemaphore> imported{};
};

struct GPUFrameResources {
    const VSAPI* vsapi;
    std::vector<const VSFrame*> frames;
    std::vector<std::shared_ptr<GPUMemoryImport>> memory;
    std::vector<std::shared_ptr<GPUSemaphoreImport>> semaphores;
    ~GPUFrameResources() {
        for (auto frame : frames)
            vsapi->freeFrame(frame);
    }
};

struct GPUInterop {
    const VSAPI* vsapi;
    const VSVULKANAPI* vkapi;
    VSCore* core;
    VSVulkanCoreInfo info{};
    std::mutex lock;
    std::vector<std::shared_ptr<GPUMemoryImport>> memory;
    std::vector<std::shared_ptr<GPUSemaphoreImport>> semaphores;

    GPUInterop(const VSAPI* api, VSCore* c) : vsapi(api), vkapi(api->getVulkanAPI()), core(c) {}

    std::optional<std::string> init(int& device_id) {
        char error[512]{};
        if (vkapi->getVulkanCoreInfo(core, &info, error, sizeof(error)))
            return error;
        if (!info.exportHandleType)
            return "the input Vulkan device cannot export GPU memory to CUDA";
        if (!info.semaphoreExportHandleType)
            return "the input Vulkan device cannot export timeline semaphores to CUDA";
        return selectCUDADevice(&info, device_id, true);
    }

    std::optional<std::string> mapPlane(const VSFrame* frame, int plane, uint8_t*& out, GPUFrameResources& resources) {
        auto set_error = [](const std::string& error) { return error; };
        char error[512]{};
        VSVulkanExportedMemory exp{};
        if (vkapi->exportGPUPlane(frame, plane, &exp, error, sizeof(error)))
            return error;
        GPUExportHandle handle{exp.handle};
        std::lock_guard<std::mutex> guard{lock};
        auto it =
            std::find_if(memory.begin(), memory.end(), [&](const auto& entry) { return entry->id == exp.memoryId; });
        std::shared_ptr<GPUMemoryImport> entry;
        if (it != memory.end()) {
            entry = *it;
            memory.erase(it);
        } else {
            entry = std::make_shared<GPUMemoryImport>();
            entry->id = exp.memoryId;
            entry->vkapi = vkapi;
            if (VS_GPU_IMPORT_CHARGED_AGAIN) {
                entry->reservation =
                    vkapi->reserveGPUMemory(core, static_cast<int64_t>(exp.memorySize), error, sizeof(error));
                if (!entry->reservation)
                    return error;
            }
            cudaExternalMemoryHandleDesc desc{};
#ifdef _WIN32
            desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
            desc.handle.win32.handle = reinterpret_cast<void*>(exp.handle);
#else
            desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
            desc.handle.fd = static_cast<int>(exp.handle);
#endif
            desc.size = exp.memorySize;
            checkError(cudaImportExternalMemory(&entry->memory.data, &desc));
            handle.imported();
            cudaExternalMemoryBufferDesc buffer{};
            buffer.size = exp.memorySize;
            checkError(
                cudaExternalMemoryGetMappedBuffer(reinterpret_cast<void**>(&entry->base.data), entry->memory, &buffer)
            );
        }
        memory.push_back(entry);
        // in flight submissions retain their own references, eviction never invalidates queued work
        if (memory.size() > 8)
            memory.erase(memory.begin());
        resources.memory.push_back(entry);
        out = entry->base.data + exp.offset;
        return {};
    }

    std::optional<std::string> mapSemaphore(VkSemaphore semaphore, std::shared_ptr<GPUSemaphoreImport>& out) {
        auto set_error = [](const std::string& error) { return error; };
        std::lock_guard<std::mutex> guard{lock};
        auto it = std::find_if(semaphores.begin(), semaphores.end(), [&](const auto& entry) {
            return entry->semaphore == semaphore;
        });
        if (it != semaphores.end()) {
            out = *it;
            semaphores.erase(it);
            semaphores.push_back(out);
            return {};
        }
        char error[512]{};
        VSVulkanExportedSemaphore exp{};
        if (vkapi->exportGPUSemaphore(core, semaphore, &exp, error, sizeof(error)))
            return error;
        GPUExportHandle handle{exp.handle};
        auto entry = std::make_shared<GPUSemaphoreImport>();
        entry->semaphore = semaphore;
        cudaExternalSemaphoreHandleDesc desc{};
#ifdef _WIN32
        desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
        desc.handle.win32.handle = reinterpret_cast<void*>(exp.handle);
#else
        desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
        desc.handle.fd = static_cast<int>(exp.handle);
#endif
        checkError(cudaImportExternalSemaphore(&entry->imported.data, &desc));
        handle.imported();
        // exportable inputs name the core's transfer timelines, which outlive this filter
        semaphores.push_back(entry);
        if (semaphores.size() > 16)
            semaphores.erase(semaphores.begin());
        out = std::move(entry);
        return {};
    }
};

struct GPUStream {
    static constexpr size_t kMaxPending = 8;
    GPUInterop& interop;
    cudaStream_t stream;
    VSGPUTimeline* timeline{};
    std::shared_ptr<GPUSemaphoreImport> producer;
    uint64_t next_value{};
    struct Pending {
        EventResource done;
        std::unique_ptr<GPUFrameResources> resources;
    };
    std::deque<Pending> pending;

    GPUStream(GPUInterop& i, cudaStream_t s) : interop(i), stream(s) {}
    ~GPUStream() {
        cudaStreamSynchronize(stream);
        pending.clear();
        producer.reset();
        if (timeline)
            interop.vkapi->freeGPUTimeline(timeline);
    }

    std::optional<std::string> init() {
        char error[512]{};
        timeline = interop.vkapi->createGPUTimeline(interop.core, error, sizeof(error));
        if (!timeline)
            return error;
        return interop.mapSemaphore(interop.vkapi->getGPUTimelineSemaphore(timeline), producer);
    }

    std::optional<std::string> reap() {
        auto set_error = [](const std::string& error) { return error; };
        while (!pending.empty()) {
            auto result = cudaEventQuery(pending.front().done);
            if (result == cudaErrorNotReady)
                break;
            checkError(result);
            pending.pop_front();
        }
        if (pending.size() >= kMaxPending) {
            checkError(cudaEventSynchronize(pending.front().done));
            pending.pop_front();
        }
        return {};
    }

    std::optional<std::string> waitInputs(GPUFrameResources& resources) {
        auto set_error = [](const std::string& error) { return error; };
        std::vector<uint64_t> values;
        for (auto frame : resources.frames) {
            auto format = interop.vsapi->getVideoFrameFormat(frame);
            for (int p = 0; p < format->numPlanes; ++p) {
                VSVulkanPlaneInfo info{};
                interop.vkapi->getGPUPlane(frame, p, &info);
                if (!info.readySemaphore)
                    continue;
                auto it =
                    std::find_if(resources.semaphores.begin(), resources.semaphores.end(), [&](const auto& entry) {
                        return entry->semaphore == info.readySemaphore;
                    });
                if (it != resources.semaphores.end()) {
                    auto index = static_cast<size_t>(it - resources.semaphores.begin());
                    values[index] = std::max(values[index], info.readyValue);
                    continue;
                }
                std::shared_ptr<GPUSemaphoreImport> entry;
                if (auto err = interop.mapSemaphore(info.readySemaphore, entry))
                    return err;
                resources.semaphores.push_back(std::move(entry));
                values.push_back(info.readyValue);
            }
        }
        for (size_t i = 0; i < values.size(); ++i) {
            cudaExternalSemaphoreWaitParams params{};
            params.params.fence.value = values[i];
            checkError(cudaWaitExternalSemaphoresAsync(&resources.semaphores[i]->imported.data, &params, 1, stream));
        }
        return {};
    }

    std::optional<std::string>
    finish(std::unique_ptr<GPUFrameResources>& resources, const std::vector<VSFrame*>& outputs) {
        auto set_error = [](const std::string& error) { return error; };
        EventResource done{};
        checkError(cudaEventCreateWithFlags(&done.data, cudaEventDisableTiming));
        auto value = ++next_value;
        cudaExternalSemaphoreSignalParams params{};
        params.params.fence.value = value;
        checkError(cudaSignalExternalSemaphoresAsync(&producer->imported.data, &params, 1, stream));
        checkError(cudaEventRecord(done, stream));
        // enqueue the signal before publishing
        for (auto frame : outputs) {
            auto format = interop.vsapi->getVideoFrameFormat(frame);
            for (int p = 0; p < format->numPlanes; ++p)
                interop.vkapi->setGPUPlaneProducer(frame, p, timeline, value);
        }
        pending.push_back({std::move(done), std::move(resources)});
        return {};
    }
};

#endif // VSTRT_GPU_HELPER_H_
