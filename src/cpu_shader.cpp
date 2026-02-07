#include "cpu_shader.h"

static ShaderInput global_input;

void set_global_shader_input(const ShaderInput &input) {
    global_input = input;
}

const ShaderInput &get_global_shader_input() {
    return global_input;
}
