cmake_minimum_required(VERSION 4.1)
project(cpu_shader)

option(USE_CUDA "build cuda acceleration app" ON)

enable_language(CXX)

if (USE_CUDA)
    enable_language(CUDA)

    set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr --extended-lambda -diag-suppress 1394")
endif ()

set(CMAKE_CXX_STANDARD 23)

find_package(Eigen3 CONFIG REQUIRED)
find_package(SFML COMPONENTS Network Graphics Window Audio System CONFIG REQUIRED)
find_package(ImGui-SFML CONFIG REQUIRED)
find_package(Stb REQUIRED)
find_package(TBB CONFIG REQUIRED)

set(CPU_SHADER_SRC src/cpu_shader.cpp)
set(CUDA_SHADER_SRC src/run_on_cuda.cu)

set(CPU_SHADER_ENTRY_SRC src/entries/cpu_shader_entry.cpp)
set(CUDA_SHADER_ENTRY_SRC src/entries/cuda_shader_entry.cu)

add_library(cpu_shader_lib STATIC ${CPU_SHADER_ENTRY_SRC})
target_link_libraries(cpu_shader_lib PUBLIC Eigen3::Eigen)
target_link_libraries(cpu_shader_lib PUBLIC TBB::tbb)
target_include_directories(cpu_shader_lib PUBLIC include)

add_executable(cpu_shader_img src/bin/img.cpp ${CPU_SHADER_SRC})
target_link_libraries(cpu_shader_img PRIVATE cpu_shader_lib)
target_link_libraries(cpu_shader_img PRIVATE Eigen3::Eigen)
target_include_directories(cpu_shader_img PRIVATE ${Stb_INCLUDE_DIR})
target_link_libraries(cpu_shader_img PRIVATE TBB::tbb)
target_include_directories(cpu_shader_img PRIVATE include)

add_executable(cpu_shader_window src/bin/window.cpp ${CPU_SHADER_SRC})
target_link_libraries(cpu_shader_window PRIVATE cpu_shader_lib)
target_link_libraries(cpu_shader_window PRIVATE Eigen3::Eigen)
target_link_libraries(cpu_shader_window PRIVATE SFML::Network SFML::Graphics SFML::Window SFML::Audio SFML::System)
target_link_libraries(cpu_shader_window PRIVATE ImGui-SFML::ImGui-SFML)
target_link_libraries(cpu_shader_window PRIVATE TBB::tbb)
target_include_directories(cpu_shader_window PRIVATE include)


if (USE_CUDA)
    add_library(cuda_shader_lib STATIC ${CUDA_SHADER_ENTRY_SRC})
    target_link_libraries(cuda_shader_lib PUBLIC Eigen3::Eigen)
    target_link_libraries(cuda_shader_lib PUBLIC TBB::tbb)
    set_target_properties(cuda_shader_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(cuda_shader_lib PUBLIC include)

    add_executable(cuda_shader_img src/bin/img_cuda.cu ${CPU_SHADER_SRC} ${CUDA_SHADER_SRC})
    target_link_libraries(cuda_shader_img PRIVATE cuda_shader_lib)
    target_link_libraries(cuda_shader_img PRIVATE Eigen3::Eigen)
    target_include_directories(cuda_shader_img PRIVATE ${Stb_INCLUDE_DIR})
    target_link_libraries(cuda_shader_img PRIVATE TBB::tbb)
    set_target_properties(cuda_shader_img PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(cuda_shader_img PRIVATE include)

    add_executable(cuda_shader_window src/bin/window_cuda.cu ${CPU_SHADER_SRC} ${CUDA_SHADER_SRC})
    target_link_libraries(cuda_shader_window PRIVATE cuda_shader_lib)
    target_link_libraries(cuda_shader_window PRIVATE Eigen3::Eigen)
    target_link_libraries(cuda_shader_window PRIVATE SFML::Network SFML::Graphics SFML::Window SFML::Audio SFML::System)
    target_link_libraries(cuda_shader_window PRIVATE ImGui-SFML::ImGui-SFML)
    target_link_libraries(cuda_shader_window PRIVATE TBB::tbb)
    set_target_properties(cuda_shader_window PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(cuda_shader_window PRIVATE include)

endif ()

