#ifndef CPU_SHADER_H
#define CPU_SHADER_H

#include <functional>
#include <chrono>
#include <array>
#include <vector>
#include <eigen3/Eigen/Eigen>
#include <tbb/tbb.h>

struct ShaderInput {
    Eigen::Vector2f Resolution;
    Eigen::Vector4f Mouse;
    std::chrono::duration<float> Seconds;
    std::tm Date;
    std::array<float, 6> UserData;
};

using shader_func = std::function<Eigen::Vector4f(Eigen::Vector2f, const ShaderInput &input)>;
using cuda_shader_func = Eigen::Vector4f(*)(Eigen::Vector2f, const ShaderInput &input);


#if defined(__CUDACC__)
#define SHADER_FUNC __device__
#else
#define SHADER_FUNC
#endif

extern SHADER_FUNC shader_func cpu_shader_entry;
extern SHADER_FUNC cuda_shader_func cuda_shader_entry;
//data[row*width+col]
inline void run_shader(const ShaderInput &input, std::vector<Eigen::Vector4f> &data_out, const shader_func &func) {
    tbb::parallel_for(
            tbb::blocked_range2d<size_t>(0, std::lround(input.Resolution.y()), 0, std::lround(input.Resolution.x())),
            [&](tbb::blocked_range2d<size_t> r) {
                for (size_t x = r.cols().begin(); x < r.cols().end(); ++x) {
                    for (size_t y = r.rows().begin(); y < r.rows().end(); ++y) {
                        data_out[y * input.Resolution.x() + x] = func(
                                Eigen::Vector2f{x, input.Resolution.y() - y - 1},
                                input
                                );
                    }
                }
            }
            );
}


inline std::vector<std::uint8_t> float_to_data(const std::vector<Eigen::Vector4f> &data) {
    std::vector<std::uint8_t> out_data(data.size() * 4);
    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, data.size(),
                                       std::max(2048ull, out_data.size() / (std::thread::hardware_concurrency() * 32))),
            [&](const tbb::blocked_range<size_t> &r) {
                for (size_t i = r.begin(); i < r.end(); ++i) {
                    out_data[i * 4 + 0] = std::clamp<std::uint8_t>(
                            data[i].x() * std::numeric_limits<std::uint8_t>::max(), 0,
                            std::numeric_limits<std::uint8_t>::max());
                    out_data[i * 4 + 1] = std::clamp<std::uint8_t>(
                            data[i].y() * std::numeric_limits<std::uint8_t>::max(), 0,
                            std::numeric_limits<std::uint8_t>::max());
                    out_data[i * 4 + 2] = std::clamp<std::uint8_t>(
                            data[i].z() * std::numeric_limits<std::uint8_t>::max(), 0,
                            std::numeric_limits<std::uint8_t>::max());
                    out_data[i * 4 + 3] = std::clamp<std::uint8_t>(
                            data[i].w() * std::numeric_limits<std::uint8_t>::max(), 0,
                            std::numeric_limits<std::uint8_t>::max());
                }
            });
    return out_data;
}

inline void float_to_data(const std::vector<Eigen::Vector4f> &data, std::vector<std::uint8_t> &out_data) {
    out_data.resize(data.size() * 4);
    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, data.size(),
                                       std::max(2048ull, out_data.size() / (std::thread::hardware_concurrency() * 32))),
            [&](const tbb::blocked_range<size_t> &r) {
                for (size_t i = r.begin(); i < r.end(); ++i) {
                    out_data[i * 4 + 0] = std::clamp<std::uint8_t>(
                            data[i].x() * std::numeric_limits<std::uint8_t>::max(), 0,
                            std::numeric_limits<std::uint8_t>::max());
                    out_data[i * 4 + 1] = std::clamp<std::uint8_t>(
                            data[i].y() * std::numeric_limits<std::uint8_t>::max(), 0,
                            std::numeric_limits<std::uint8_t>::max());
                    out_data[i * 4 + 2] = std::clamp<std::uint8_t>(
                            data[i].z() * std::numeric_limits<std::uint8_t>::max(), 0,
                            std::numeric_limits<std::uint8_t>::max());
                    out_data[i * 4 + 3] = std::clamp<std::uint8_t>(
                            data[i].w() * std::numeric_limits<std::uint8_t>::max(), 0,
                            std::numeric_limits<std::uint8_t>::max());
                }
            });
}

void set_global_shader_input(const ShaderInput &input);
const ShaderInput &get_global_shader_input();

#endif //CPU_SHADER_H
