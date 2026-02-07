//
// Created by 91246 on 2026/2/7.
//

#ifndef CPU_SHADER_RUN_ON_CUDA_H
#define CPU_SHADER_RUN_ON_CUDA_H

#include "cpu_shader.h"
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                   \
    do                                                    \
    {                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
    printf("CUDA Error:\n");                      \
    printf("    File:       %s\n", __FILE__);     \
    printf("    Line:       %d\n", __LINE__);     \
    printf("    Error code: %d\n", error_code);   \
    printf("    Error text: %s\n",                \
    cudaGetErrorString(error_code));          \
    exit(1);                                      \
    }                                                 \
    } while (0)

void run_shader_on_cuda(const ShaderInput &input, void *d_data_out,
                        cuda_shader_func device_func_ptr, bool out_bytes = false);


#endif //CPU_SHADER_RUN_ON_CUDA_H
