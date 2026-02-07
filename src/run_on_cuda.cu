#include "run_on_cuda.h"
#include <iostream>
#include <stdexcept>

__global__ void cuda_entry_for_float(ShaderInput input, size_t pixel_count, Eigen::Vector2i resolution,
                                     Eigen::Vector4f *data_out, cuda_shader_func func) {

    auto start_x = threadIdx.x + blockIdx.x * blockDim.x;
    if (start_x >= pixel_count)[[unlikely]] {
        return;
    }

    size_t x = start_x % resolution.x();
    size_t y = start_x / resolution.x();

    data_out[start_x] = func(
            Eigen::Vector2f{x, input.Resolution.y() - y - 1},
            input
            );
}

__global__ void cuda_entry_for_uint8(ShaderInput input, size_t pixel_count, Eigen::Vector2i resolution,
                                     std::uint8_t *data_out, cuda_shader_func func) {

    auto start_x = threadIdx.x + blockIdx.x * blockDim.x;
    if (start_x >= pixel_count)[[unlikely]] {
        return;
    }

    size_t x = start_x % resolution.x();
    size_t y = start_x / resolution.x();

    auto ret = func(
            Eigen::Vector2f{x, input.Resolution.y() - y - 1},
            input
            );

    data_out[4 * start_x + 0] = std::lround(ret.x() * 255.);
    data_out[4 * start_x + 1] = std::lround(ret.y() * 255.);
    data_out[4 * start_x + 2] = std::lround(ret.z() * 255.);
    data_out[4 * start_x + 3] = std::lround(ret.w() * 255.);
}

void run_shader_on_cuda(const ShaderInput &input, void *d_data_out,
                        cuda_shader_func device_func_ptr, bool out_bytes) {
    size_t pixel_count = input.Resolution.x() * input.Resolution.y();

    int blockSize = 256 + 128;
    int numBlock = pixel_count / blockSize + 1;


    Eigen::Vector2i resolution = input.Resolution.cast<int>();

    if (!out_bytes)
        cuda_entry_for_float<<<numBlock,blockSize>>>(input, pixel_count, resolution,
                                                     static_cast<Eigen::Vector4f *>(d_data_out), device_func_ptr);
    else
        cuda_entry_for_uint8<<<numBlock,blockSize>>>(input, pixel_count, resolution,
                                                     static_cast<std::uint8_t *>(d_data_out),
                                                     device_func_ptr);

    auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error(cudaGetErrorString(error));
        return;
    }

    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error(cudaGetErrorString(error));
        return;
    }
}
