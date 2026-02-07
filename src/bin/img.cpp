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

int main() {
    constexpr size_t width = 1920;
    constexpr size_t height = 1080;

    ShaderInput input{
            .Resolution = Eigen::Vector2f(width, height),
            .Mouse = Eigen::Vector4f(width, height, 0, 0),
            .Seconds = std::chrono::duration<float>(0),
            .Date = get_tm()
    };
    std::vector<Eigen::Vector4f> data(width * height);

    auto begin = std::chrono::high_resolution_clock::now();
    run_shader(input, data, cpu_shader_entry);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

    std::cout << "shader runs for:" << duration.count() << "ms" << std::endl;


    auto data_in_bytes = float_to_data(data);
    stbi_write_png("output.png", width, height, 4, data_in_bytes.data(), width * 4);

    return 0;
}
