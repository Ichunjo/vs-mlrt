#ifdef _MSC_VER
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>

#include <delayimp.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(TRT_MAJOR_RTX)
#define DLL_DIR L"../../../tensorrt_rtx_libs"
#else
#define DLL_DIR L"../../../tensorrt_libs"
#endif

#include <NvInferRuntime.h>
#include <iostream>

#define TO_STRING(x) #x
#define CONCAT_VERSION(name, version) (name "_" TO_STRING(version) ".dll")
#define CONCAT_VERSION2(name, major, minor) (name "_" TO_STRING(major) "_" TO_STRING(minor) ".dll")

namespace {
std::vector<std::wstring> dlls = {
// This list must be sorted by dependency.
#if defined(TRT_MAJOR_RTX)
    CONCAT_VERSION2(L"tensorrt_rtx", TRT_MAJOR_RTX, TRT_MINOR_RTX),
#else
#ifdef USE_NVINFER_PLUGIN
    // nvinfer_plugin dependencies
    CONCAT_VERSION(L"nvinfer", NV_TENSORRT_MAJOR),
    CONCAT_VERSION(L"nvinfer_plugin", NV_TENSORRT_MAJOR),
#endif // USE_NVINFER_PLUGIN
       // Finally, nvinfer again.
    CONCAT_VERSION(L"nvinfer", NV_TENSORRT_MAJOR), // must be the last
#endif
};

namespace fs = std::filesystem;
static fs::path dllDir() {
    static const std::wstring res = []() -> std::wstring {
        HMODULE mod = 0;
        if (GetModuleHandleExA(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                (char*)dllDir,
                &mod
            )) {
            std::vector<wchar_t> buf;
            size_t n = 0;
            do {
                buf.resize(buf.size() + MAX_PATH);
                n = GetModuleFileNameW(mod, buf.data(), static_cast<DWORD>(buf.size()));
            } while (n >= buf.size());
            buf.resize(n);
            std::wstring path(buf.begin(), buf.end());
            return path;
        }
        throw std::runtime_error("unable to locate myself");
    }();
    return fs::path(res).parent_path();
}

FARPROC loadDLLs() {
    fs::path dir = dllDir() / DLL_DIR;
    HMODULE h = nullptr;
    for (const auto dll : dlls) {
        fs::path p = dir / dll;
        std::wstring s = p;
        h = LoadLibraryW(s.c_str());
        DWORD err = GetLastError();
        if (getenv("VSTRT_VERBOSE"))
            std::wcerr << L"vstrt: preloading " << p << L": " << h << std::endl;
        if (!h)
            std::wcerr << L"vstrt: failed to preload " << s << L", errno " << err << std::endl;
    }
    return (FARPROC)h;
}

static int dummy() { // mimic getInferLibVersion
    return 0;
}

extern "C" FARPROC WINAPI delayload_hook(unsigned reason, DelayLoadInfo* info) {
    switch (reason) {
    case dliNoteStartProcessing:
    case dliNoteEndProcessing:
        // Nothing to do here.
        break;
    case dliNotePreLoadLibrary:
        // std::cerr << "loading " << info->szDll << std::endl;
        loadDLLs();
        return (FARPROC)LoadLibraryA(info->szDll);
    case dliNotePreGetProcAddress:
        // Nothing to do here.
        break;
    case dliFailLoadLib:
    case dliFailGetProc:
        // Returning NULL from error notifications will cause the delay load
        // runtime to raise a VcppException structured exception, that some code
        // might want to handle.
        // return NULL;
        // The SE will crash the process, so instead we return a dummy function.
        return (FARPROC)dummy;
        break;
    default:
        abort(); // unreachable.
        break;
    }
    // Returning NULL causes the delay load machinery to perform default
    // processing for this notification.
    return NULL;
}
} // namespace

extern "C" {
const PfnDliHook __pfnDliNotifyHook2 = delayload_hook;
const PfnDliHook __pfnDliFailureHook2 = delayload_hook;
};
#endif // _MSC_VER
