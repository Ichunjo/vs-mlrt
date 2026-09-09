#ifndef VSTRT_CUDA_HELPER_H_
#define VSTRT_CUDA_HELPER_H_

#include <cuda_runtime_api.h>
#include <string>

#define checkError(expr)                                                                                               \
    do {                                                                                                               \
        using namespace std::string_literals;                                                                          \
        cudaError_t __err = expr;                                                                                      \
        if (__err != cudaSuccess) {                                                                                    \
            const char* message = cudaGetErrorString(__err);                                                           \
            return set_error("'"s + #expr + "' failed: " + message);                                                   \
        }                                                                                                              \
    } while (0)

#endif // VSTRT_CUDA_HELPER_H_
