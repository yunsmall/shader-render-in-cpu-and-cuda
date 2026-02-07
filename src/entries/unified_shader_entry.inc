#include <eigen3/Eigen/Eigen>
#include "utils.h"

#include <numbers>
#include <ranges>

SHADER_FUNC Eigen::Vector4f move_circle_shader(Eigen::Vector2f fragCoord, const ShaderInput &input) {
    Eigen::Vector2f uv = to_uv_center(fragCoord, input);

    // Eigen::Vector4f color{1, 1, 1, 1};
    Eigen::Vector4f color{0, 0, 0, 1};
    [[maybe_unused]] constexpr float color_T = 10;

    if (input.Mouse.z() > 0) {
        auto mouse_p = to_uv_center(input.Mouse.topRows(2), input);
        auto d = (uv - mouse_p).norm();

        // Eigen::Vector4f circle_color{(1 + sin(2 * std::numbers::pi_v<float> / color_T * input.Seconds.count())) * 0.5f,
        //                              0, 0,
        //                              smoothstep(0.1f, 0.05f, d)};
        Eigen::Vector4f circle_color{1, 0, 0,
                                     smoothstep(0.1f, 0.05f, d)};
        color = mix(color, circle_color, circle_color.w()).eval();

        Eigen::Vector2f rect_size{0.05, 0.03};
        Eigen::Vector4f rect_color{0, 0, 1,
                                   bond<Eigen::Vector2f>(
                                           -rect_size + mouse_p,
                                           -rect_size * 0.8 + mouse_p,
                                           rect_size + mouse_p,
                                           rect_size * 0.8 + mouse_p,
                                           uv
                                           )
        };
        color = mix(color, rect_color, rect_color.w()).eval();
    }
    return color;
}

SHADER_FUNC Eigen::Vector4f rgb_led_shader(Eigen::Vector2f fragCoord, const ShaderInput &input) {
    Eigen::Vector2f uv = to_uv(fragCoord, input);

    Eigen::Vector4f color{0, 0, 0, 1};

    constexpr float color_div_rate = 1000.;
    int color_index = int(floor(uv.x() * color_div_rate)) % 7;

    Eigen::Vector4f led_color = Eigen::Vector4f(0., 0., 0., 1.);

    if (color_index >= 0 && color_index <= 1) {
        led_color.x() = 1.;
    }
    else if (color_index >= 2 && color_index <= 3) {
        led_color.y() = 1.;
    }
    else if (color_index >= 4 && color_index <= 5) {
        led_color.z() = 1.;
    }

    color = led_color;

    constexpr float vertical_div_rate = color_div_rate;
    int vertical_black_index = int(floor(uv.y() * vertical_div_rate)) % 7;

    if (vertical_black_index == 0) {
        color = Eigen::Vector4f(0, 0, 0, 1);
    }

    return color;
}

SHADER_FUNC Eigen::Vector4f circles_shader(Eigen::Vector2f fragCoord, const ShaderInput &input) {
    Eigen::Vector2f uv = to_uv_center(fragCoord, input);

    Eigen::Vector3f eye_position{0, 0, 5};
    float screen_distance = 2;
    Eigen::Vector3f screen_direction = Eigen::Vector3f{cos(input.Seconds.count()), 0, sin(input.Seconds.count())};
    Eigen::Vector3f screen_r = screen_direction.normalized() * screen_distance;

    Eigen::Vector3f pixel_direction = screen_r + Eigen::Vector3f{uv.x(), uv.y(), 0};

    // Eigen::Vector3f screen_position = eye_position + pixel_direction;


    Eigen::Vector4f color{0, 0, 0, 1};

    Eigen::Vector3f cube_points[] = {
            {0, 0, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 1, 1},
            {1, 0, 0},
            {1, 0, 1},
            {1, 1, 0},
            {1, 1, 1},
    };

    for (auto &point: cube_points) {
        if ((point - eye_position).dot(pixel_direction) > 0) {
            auto hit = smoothstep(0.06f, 0.05f, DistToPoint(eye_position, pixel_direction, point));
            color += Eigen::Vector4f{hit, hit, hit, 0};
        }
    }


    return color;
}

SHADER_FUNC Eigen::Vector4f mix1_shader(Eigen::Vector2f fragCoord, const ShaderInput &input) {
    return (move_circle_shader(fragCoord, input).array() * rgb_led_shader(fragCoord, input).array()).matrix();
}

SHADER_FUNC Eigen::Vector4f ray_marching_test_shader(Eigen::Vector2f fragCoord, const ShaderInput &input) {
    Eigen::Vector2f uv = to_uv_center(fragCoord, input);
    Eigen::Vector4f color{0, 0, 0, 1};

    Eigen::Vector2f m = to_uv({input.Mouse.x(), input.Mouse.y()}, input);

    // Eigen::Vector3f ro{2, 2, 2};
    // Eigen::Vector3f ro = euler(2 * std::numbers::pi_v<float> / 5 * input.Seconds.count(),
    //                            std::numbers::pi_v<float> * 45 / 180, {0, 1, 0}, {1, 0, 0}) * 3;

    Eigen::Vector3f ro = spherical_coordinates(2 * std::numbers::pi_v<float> * m.x(),
                                               std::numbers::pi_v<float> * m.y(),
                                               {0, 1, 0},
                                               {1, 0, 0}) * 3;

    Eigen::Vector3f up{0, 1, 0};

    auto rd = get_ray_dir(uv, ro, {0, 0, 0}, up, 1);

    get_dist_func get_dist1 = [](const Eigen::Vector3f &p, void *user_data)-> float {
        const ShaderInput &input = *((const ShaderInput *) user_data);
        return template_min(
                // sdCircle(p, {1, 0, 0}, 0.5),
                // sdCircle(p, {-1, 0, 0}, 0.5),
                // sdBox(p,
                //       {2.f * sin(input.Seconds.count()), 0, 0},
                //       {0.2, 0.2, 0.2},
                //       Eigen::AngleAxisf{
                //               std::numbers::pi_v<float> * 45 / 180,
                //               Eigen::Vector3f{0, 1, 1}.normalized()
                //       }.
                //       toRotationMatrix()),
                mix(sdCircle(p, {0, 0, 1}, 0.5),
                    sdBox(p,
                          {0, 0, 1},
                          {0.2, 0.2, 0.2}
                            ),
                    0.5 * sin(input.Seconds.count() * input.UserData[0]) + 0.5),
                smin(0.2f,
                     sdCircle(p, {1, 0, 0}, 0.3),
                     sdBox(p,
                           {0, 0, 0},
                           {0.2, 0.2, 0.2}
                             ),
                     // sdCircle(p, {0, 0, 0}, 0.3),
                     sdCircle(p, {-1, 0, 0}, 0.3),
                     sdBox(p,
                           {2.f * sin(input.Seconds.count() / 6), 0, 0},
                           {0.2, 0.2, 0.2},
                           Eigen::AngleAxisf{
                                   std::numbers::pi_v<float> * 45 / 180,
                                   Eigen::Vector3f{0, 1, 1}.normalized()
                           }.
                           toRotationMatrix())
                        ),
                sdCapsule(p, {-1, -1, 0}, {1, -1, 0}, 0.1),
                sdPlane(p, {0, -2, 0}, {0, 1, 0}),
                sdCylinder(p, {0, -1, -1}, {0, -1, 1}, 0.1)
                );
        // float min_value = std::numeric_limits<float>::max();
        // for (auto &sdf: objs) {
        //     if (sdf < min_value) {
        //         min_value = sdf;
        //     }
        // }
        // return min_value;
    };

    get_dist_func get_dist = [](const Eigen::Vector3f &p, void *user_data)-> float {
        // return gyroid(p);
        return std::max(sdBox(p, {0, 0, 0}, {1, 1, 1}), gyroid(p, 10));
    };

    void *user_data = (void *) &input;
    constexpr float max_dist = 100.f;

    // printf("uv %f,%f begin ray_marching\n", uv.x(), uv.y());
    float d = ray_marching(ro, rd, get_dist, user_data, max_dist);
    // printf("uv %f,%f end ray_marching\n", uv.x(), uv.y());

    if (d < max_dist) {
        auto p = ro + rd * d;
        float dif = get_light(p, {0, 5, 5}, get_dist, user_data, max_dist);
        color = Eigen::Vector4f(dif, dif, dif, 1.0);
        color.topRows(3).array() *= Eigen::Vector3f{1, 1, 1}.array();
    }

    // cuda里为什么不能这么写？？？？？？？？？
    // color.topRows<3>() = color.topRows<3>().array().pow(0.4545f).matrix().eval();
    color.topRows<3>() = color.topRows<3>().unaryExpr([](float v) {
        return powf(v, 0.4545f);
    }).eval();
    return color;
}

SHADER_FUNC Eigen::Vector4f mix2_shader(Eigen::Vector2f fragCoord, const ShaderInput &input) {
    return (ray_marching_test_shader(fragCoord, input).array() * rgb_led_shader(fragCoord, input).array()).matrix();
}

#if defined(__CUDACC__)
SHADER_FUNC cuda_shader_func cuda_shader_entry = ray_marching_test_shader;
#else
SHADER_FUNC shader_func cpu_shader_entry = ray_marching_test_shader;
#endif
