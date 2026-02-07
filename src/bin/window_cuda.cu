#include <iostream>
#include <functional>
#include <chrono>
#include <array>
#include <vector>
#include <numbers>

#include <eigen3/Eigen/Eigen>
#include <tbb/tbb.h>
#include <SFML/Graphics.hpp>
#include <imgui.h>
#include <imgui-SFML.h>

#include "cpu_shader.h"
#include "utils.h"
#include "run_on_cuda.h"

#include <cuda_gl_interop.h>


int copy_cuda_data_to_texture(void *data, size_t count, sf::Texture &texture) {
    // 错误检查宏
#define CHECK_CUDA_ERROR(err) \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (%s:%d)\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            return -1; \
        }

    // 获取纹理尺寸
    sf::Vector2u textureSize = texture.getSize();
    unsigned int width = textureSize.x;
    unsigned int height = textureSize.y;

    // 验证数据大小是否匹配纹理尺寸
    size_t expectedCount = width * height * 4; // RGBA = 4 channels
    if (count != expectedCount) {
        fprintf(stderr, "Error: Data size mismatch. Expected %zu elements, got %zu\n",
                expectedCount, count);
        return -2;
    }

    // 获取 OpenGL 纹理 ID
    unsigned int textureID = texture.getNativeHandle();
    if (textureID == 0) {
        fprintf(stderr, "Error: Invalid texture handle\n");
        return -3;
    }

    cudaError_t cudaErr = cudaSuccess;
    cudaGraphicsResource *cudaResource = nullptr;

    // 1. 注册 OpenGL 纹理为 CUDA 资源
    cudaErr = cudaGraphicsGLRegisterImage(&cudaResource, textureID,
                                          GL_TEXTURE_2D,
                                          cudaGraphicsRegisterFlagsWriteDiscard);
    if (cudaErr != cudaSuccess) {
        fprintf(stderr, "Failed to register texture with CUDA: %s\n",
                cudaGetErrorString(cudaErr));
        return -4;
    }

    // 2. 映射 CUDA 资源
    cudaErr = cudaGraphicsMapResources(1, &cudaResource, 0);
    CHECK_CUDA_ERROR(cudaErr);

    // 3. 获取 CUDA 数组
    cudaArray *cuArray = nullptr;
    cudaErr = cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaResource, 0, 0);
    CHECK_CUDA_ERROR(cudaErr);

    // 4. 将设备内存数据复制到 CUDA 数组
    cudaMemcpy3DParms copyParams = {0};

    // 设置源设备内存
    copyParams.srcPtr = make_cudaPitchedPtr(data, width * 4 * sizeof(uint8_t),
                                            width, height);

    // 设置目标 CUDA 数组
    copyParams.dstArray = cuArray;

    // 设置复制类型：设备到数组
    copyParams.kind = cudaMemcpyDeviceToDevice;

    // 设置复制范围
    copyParams.extent = make_cudaExtent(width, height, 1);

    // 执行复制
    cudaErr = cudaMemcpy3D(&copyParams);
    CHECK_CUDA_ERROR(cudaErr);

    // 5. 取消映射资源
    cudaErr = cudaGraphicsUnmapResources(1, &cudaResource, 0);
    CHECK_CUDA_ERROR(cudaErr);

    // 6. 注销 CUDA 资源
    cudaErr = cudaGraphicsUnregisterResource(cudaResource);
    CHECK_CUDA_ERROR(cudaErr);

    return 0;

#undef CHECK_CUDA_ERROR
}


int main() {
    constexpr size_t default_width = 800;
    constexpr size_t default_height = 600;

    // constexpr size_t default_width = 100;
    // constexpr size_t default_height = 50;

    sf::RenderWindow window(sf::VideoMode({default_width, default_height}), "cuda shader window");
    // window.setVerticalSyncEnabled(true);
    if (!ImGui::SFML::Init(window)) {
        std::cerr << "Failed to initialize ImGui!" << std::endl;
        return 1;
    }


    sf::Texture shader_texture{};
    sf::Sprite shader_sprite{shader_texture};

    if (!shader_texture.resize({default_width, default_height})) {
        std::cerr << "shader texture resize failed" << std::endl;
        return 1;
    }

    constexpr size_t byte_per_pixel = 4;
    size_t data_bytes_max_size_in_bytes = default_width * default_height * byte_per_pixel;
    size_t data_bytes_size_in_bytes = data_bytes_max_size_in_bytes;
    std::uint8_t *data = nullptr;
    auto cuda_error = cudaMalloc(
            &data, sizeof(std::remove_pointer_t<decltype(data)>) * default_width * default_height * 4);
    if (cuda_error != cudaSuccess) {
        std::cerr << cudaGetErrorString(cuda_error) << std::endl;
        return -1;
    }

    cuda_shader_func d_shader_entry = nullptr;
    cuda_error = cudaMemcpyFromSymbol(&d_shader_entry, cuda_shader_entry, sizeof(cuda_shader_func));
    // auto cuda_error = cudaGetSymbolAddress((void **) &d_func_addr, my_shader);
    if (cuda_error != cudaSuccess) {
        std::cerr << cudaGetErrorString(cuda_error) << std::endl;
        return -1;
    }

    std::tm last_tm = get_tm();

    sf::Clock last_tm_clock{};
    sf::Clock imgui_deltaClock{};
    sf::Clock elapsed_clock{};
    int32_t mouse_x = 0, mouse_y = 0;
    bool left_button_pressed = false, right_button_pressed = false;
    std::chrono::duration<float> frame_duration{1};
    decltype(ShaderInput{}.UserData) UserData{};
    while (window.isOpen()) {
        auto frame_start = std::chrono::steady_clock::now();
        while (auto event = window.pollEvent()) {
            ImGui::SFML::ProcessEvent(window, *event);
            if (ImGui::GetIO().WantCaptureMouse)[[unlikely]] {
                if (event->is<sf::Event::MouseButtonPressed>() ||
                    event->is<sf::Event::MouseButtonReleased>() ||
                    event->is<sf::Event::MouseMoved>() ||
                    event->is<sf::Event::MouseWheelScrolled>()) {
                    continue;
                }
            }
            if (ImGui::GetIO().WantCaptureKeyboard)[[unlikely]] {
                if (event->is<sf::Event::KeyPressed>() ||
                    event->is<sf::Event::KeyReleased>() ||
                    event->is<sf::Event::TextEntered>()) {
                }
            }

            if (event->is<sf::Event::Closed>()) {
                window.close();
                goto window_close;
            }
            else if (event->is<sf::Event::KeyPressed>()) {

            }
            else if (auto mouse_moved = event->getIf<sf::Event::MouseMoved>()) {
                if (left_button_pressed) {
                    mouse_x = mouse_moved->position.x;
                    mouse_y = mouse_moved->position.y;
                }
            }
            else if (auto mouse_pressed = event->getIf<sf::Event::MouseButtonPressed>()) {
                mouse_x = mouse_pressed->position.x;
                mouse_y = mouse_pressed->position.y;
                if (mouse_pressed->button == sf::Mouse::Button::Left) {
                    left_button_pressed = true;
                }
                else if (mouse_pressed->button == sf::Mouse::Button::Right) {
                    right_button_pressed = true;
                }
            }
            else if (auto mouse_released = event->getIf<sf::Event::MouseButtonReleased>()) {
                mouse_x = mouse_released->position.x;
                mouse_y = mouse_released->position.y;
                if (mouse_released->button == sf::Mouse::Button::Left) {
                    left_button_pressed = false;
                }
                else if (mouse_released->button == sf::Mouse::Button::Right) {
                    right_button_pressed = false;
                }
            }
            else if (auto resized = event->getIf<sf::Event::Resized>()) {
                size_t new_pixel_count = resized->size.x * resized->size.y;
                size_t new_byte_count = new_pixel_count * sizeof(std::remove_pointer_t<decltype(data)>) * 4;
                if (data_bytes_max_size_in_bytes < new_byte_count) {
                    cuda_error = cudaFree(data);
                    if (cuda_error != cudaSuccess) {
                        std::cerr << cudaGetErrorString(cuda_error) << std::endl;
                        goto window_close;
                    }

                    cuda_error = cudaMalloc(&data, new_byte_count);
                    if (cuda_error != cudaSuccess) {
                        std::cerr << cudaGetErrorString(cuda_error) << std::endl;
                        goto window_close;
                    }
                    data_bytes_max_size_in_bytes = new_byte_count;
                }
                data_bytes_size_in_bytes = new_byte_count;

                if (!shader_texture.resize(resized->size))[[unlikely]] {
                    std::cerr << "shader texture resize failed" << std::endl;
                    // goto window_close;
                }

                window.setView(sf::View{sf::FloatRect{
                        {0, 0},
                        sf::Vector2f{static_cast<float>(resized->size.x),
                                     static_cast<float>(resized->size.y)}
                }});
            }
        }
        ImGui::SFML::Update(window, imgui_deltaClock.restart());

        ImGui::Begin("debug");
        float fps = 1. / frame_duration.count();
        ImGui::InputFloat("fps", &fps, 0, 0, "%.5f", ImGuiInputTextFlags_ReadOnly);
        ImGui::InputFloat3("UserData1", &UserData[0]);
        ImGui::InputFloat3("UserData2", &UserData[3]);
        ImGui::End();

        if (last_tm_clock.getElapsedTime().asSeconds() > 0.9)[[unlikely]] {
            last_tm_clock.restart();
            last_tm = get_tm();
        }

        auto window_size = window.getSize();
        uint32_t window_width = window_size.x, window_height = window_size.y;
        mouse_x = std::clamp<int32_t>(mouse_x, 0, window_width - 1);
        mouse_y = std::clamp<int32_t>(mouse_y, 0, window_height - 1);
        ShaderInput input{
                .Resolution = Eigen::Vector2f(window_width, window_height),
                .Mouse = Eigen::Vector4f(mouse_x, window_height - mouse_y + 1,
                                         left_button_pressed, right_button_pressed),
                .Seconds = elapsed_clock.getElapsedTime().toDuration(),
                .Date = last_tm,
                .UserData = UserData
        };

        set_global_shader_input(input);

        // auto begin = std::chrono::high_resolution_clock::now();
        // run_shader(input, data, my_shader);
        // run_shader(input, data, circles_shader);
        run_shader_on_cuda(input, data, d_shader_entry, true);
        // auto end = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        // std::cout << "shader runs for:" << duration.count() << "ms" << std::endl;

        copy_cuda_data_to_texture(data, data_bytes_size_in_bytes, shader_texture);
        shader_sprite.setTexture(shader_texture, true);

        window.clear(sf::Color::Black);
        window.draw(shader_sprite);
        ImGui::SFML::Render(window);
        window.display();
        auto frame_end = std::chrono::steady_clock::now();
        frame_duration = std::chrono::duration_cast<decltype(frame_duration)>(frame_end - frame_start);
    }
window_close:
    ImGui::SFML::Shutdown();
    cudaFree(data);


    return 0;
}
