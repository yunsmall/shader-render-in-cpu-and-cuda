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

int main() {
    constexpr size_t default_width = 800;
    constexpr size_t default_height = 600;

    sf::RenderWindow window(sf::VideoMode({default_width, default_height}), "shader window");
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

    std::vector<Eigen::Vector4f> data(default_width * default_height);
    std::vector<std::uint8_t> data_in_bytes(default_width * default_height * 4);

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
                data.resize(resized->size.x * resized->size.y);
                data_in_bytes.resize(resized->size.x * resized->size.y * 4);

                if (!shader_texture.resize(resized->size))[[unlikely]] {
                    std::cerr << "shader texture resize failed" << std::endl;
                    goto window_close;
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
        run_shader(input, data, cpu_shader_entry);
        // auto end = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        // std::cout << "shader runs for:" << duration.count() << "ms" << std::endl;

        float_to_data(data, data_in_bytes);

        shader_texture.update(data_in_bytes.data());
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

    return 0;
}
