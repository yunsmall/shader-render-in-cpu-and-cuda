#include <iostream>
#include <functional>
#include <chrono>
#include <array>
#include <vector>
#include <eigen3/Eigen/Eigen>
#include <tbb/tbb.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "cpu_shader.h"
#include "utils.h"
#include "run_on_cuda.h"

int main() {
    constexpr size_t width = 1920;
    constexpr size_t height = 1080;

    // constexpr size_t width = 100;
    // constexpr size_t height = 50;

    size_t pixel_count = width * height;

    size_t stack_size = 0;
    cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);

    std::cout << "stack_size:" << stack_size << std::endl;
    cudaDeviceSetLimit(cudaLimitStackSize, stack_size * 1.5);

    ShaderInput input{
            .Resolution = Eigen::Vector2f(width, height),
            .Mouse = Eigen::Vector4f(width, height, 0, 0),
            .Seconds = std::chrono::duration<float>(0),
            .Date = get_tm()
    };
    std::vector<Eigen::Vector4f> data(pixel_count);

    cuda_shader_func d_func_addr = nullptr;
    auto cuda_error = cudaMemcpyFromSymbol(&d_func_addr, cuda_shader_entry, sizeof(cuda_shader_func));
    // auto cuda_error = cudaGetSymbolAddress((void **) &d_func_addr, my_shader);
    if (cuda_error != cudaSuccess) {
        std::cerr << cudaGetErrorString(cuda_error) << std::endl;
        return -1;
    }

    Eigen::Vector4f *d_data_out = nullptr;
    cuda_error = cudaMalloc(&d_data_out, sizeof(Eigen::Vector4f) * pixel_count);
    if (cuda_error != cudaSuccess) {
        std::cerr << cudaGetErrorString(cuda_error) << std::endl;
        throw std::runtime_error(cudaGetErrorString(cuda_error));
        return -1;
    }

    auto begin = std::chrono::high_resolution_clock::now();
    // run_shader(input, data, my_shader);
    run_shader_on_cuda(input, d_data_out, d_func_addr);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    std::cout << "shader runs for:" << duration.count() << "ms" << std::endl;


    cuda_error = cudaMemcpy(data.data(), d_data_out, pixel_count * sizeof(Eigen::Vector4f), cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) {
        std::cerr << cudaGetErrorString(cuda_error) << std::endl;
        cudaFree(d_data_out);
        return -1;
    }

    auto data_in_bytes = float_to_data(data);
    stbi_write_png("output_cuda.png", width, height, 4, data_in_bytes.data(), width * 4);


    cudaFree(d_data_out);
    return 0;
}
