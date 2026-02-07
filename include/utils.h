#ifndef CPU_SHADER_UTILS_H
#define CPU_SHADER_UTILS_H

#include <numeric>
#include <ctime>
#include <cmath>
#include <eigen3/Eigen/Eigen>

inline std::tm get_tm() {
    auto now = std::time(nullptr);
    std::tm result{};
    bool get_successfully = false;
#ifdef _MSC_VER
    get_successfully = (localtime_s(&result, &now) == 0);
#else
    get_successfully = (localtime_r(&now, &result) != nullptr);
#endif
    if (!get_successfully) {
        throw std::runtime_error("get localtime error");
    }
    return result;
}

#if defined(__CUDACC__)
#define UTILS_IN_CUDA __host__ __device__
#else
#define UTILS_IN_CUDA
#endif

UTILS_IN_CUDA inline Eigen::Vector2f to_uv(const Eigen::Vector2f &Coord, const ShaderInput &input) {
    Eigen::Vector2f uv = Coord.array() / input.Resolution.array();
    // uv.y() *= input.Resolution.y() / input.Resolution.x();
    uv.x() *= input.Resolution.x() / input.Resolution.y();
    return uv;
}

UTILS_IN_CUDA inline Eigen::Vector2f to_uv_center(const Eigen::Vector2f &Coord, const ShaderInput &input) {
    Eigen::Vector2f uv = Coord.array() / input.Resolution.array();
    uv -= Eigen::Vector2f{0.5, 0.5};
    // uv.y() *= input.Resolution.y() / input.Resolution.x();
    uv.x() *= input.Resolution.x() / input.Resolution.y();
    return uv;
}

UTILS_IN_CUDA inline float DistToPoint(const Eigen::Vector3f &Origin, const Eigen::Vector3f &Direction,
                                       const Eigen::Vector3f &Point) {
    return (Direction.cross(Origin - Point)).norm() / Direction.norm();
}

UTILS_IN_CUDA inline float DistToLine(const Eigen::Vector3f &Origin, const Eigen::Vector3f &Direction,
                                      const Eigen::Vector3f &OtherOrigin, const Eigen::Vector3f &OtherDirection) {
    auto vertical = Direction.cross(OtherDirection);
    return abs(vertical.dot(Origin - OtherOrigin)) / vertical.norm();
}

// ray marching

UTILS_IN_CUDA inline float sdCircle(const Eigen::Vector3f &p, const Eigen::Vector3f &o, float radius) {
    return (p - o).norm() - radius;
}

UTILS_IN_CUDA inline float sdBox(const Eigen::Vector3f &p, const Eigen::Vector3f &o, const Eigen::Vector3f &s,
                                 const Eigen::Matrix3f &rotation = Eigen::Matrix3f::Identity()) {
    auto new_p = (rotation.transpose() * (p - o)).cwiseAbs() - s;
    return (new_p.array().max(0.)).matrix().norm() + fmin(new_p.array().maxCoeff(), 0.f);
}

UTILS_IN_CUDA inline float sdPlane(const Eigen::Vector3f &p, const Eigen::Vector3f &p_in_plane,
                                   const Eigen::Vector3f &n) {
    return (p - p_in_plane).dot(n.normalized());
}

UTILS_IN_CUDA inline float sdCapsule(const Eigen::Vector3f &p, const Eigen::Vector3f &a, const Eigen::Vector3f &b,
                                     float radius) {
    auto ab = b - a;
    auto ap = p - a;

    auto t = ab.dot(ap) / ab.dot(ab);
    t = std::clamp(t, 0.f, 1.f);

    auto c = a + t * ab;
    return (p - c).norm() - radius;
}

UTILS_IN_CUDA inline float sdCylinder(const Eigen::Vector3f &p, const Eigen::Vector3f &a, const Eigen::Vector3f &b,
                                      float radius) {
    auto ab = b - a;
    auto ap = p - a;

    auto t = ab.dot(ap) / ab.dot(ab);
    // t = std::clamp(t, 0.f, 1.f);

    auto c = a + t * ab;

    auto x = (p - c).norm() - radius;
    auto y = (abs(t - .5f) - .5f) * ab.norm();
    auto e = Eigen::Vector2f{x, y}.cwiseMax(0.f).norm();
    auto i = std::min(std::max(x, y), 0.f);
    return e + i;
}

UTILS_IN_CUDA inline float gyroid(const Eigen::Vector3f &p, const float scale) {
    Eigen::Vector3f new_p = p * scale;
    return new_p.array().sin().matrix().dot(Eigen::Vector3f{new_p.z(), new_p.x(), new_p.y()}.array().cos().matrix()) /
           scale;
}

UTILS_IN_CUDA inline float sdf_union(const float a, const float b) {
    return std::min(a, b);
}

UTILS_IN_CUDA inline float sdf_intersection(const float a, const float b) {
    return std::max(a, b);
}

//扣掉后者
UTILS_IN_CUDA inline float sdf_difference(const float a, const float b) {
    return std::max(a, -b);
}

using get_dist_func = float (*)(const Eigen::Vector3f &p, void *user_data);


UTILS_IN_CUDA float inline ray_marching(const Eigen::Vector3f &ro, const Eigen::Vector3f &rd, get_dist_func get_dist,
                                        void *user_data = nullptr, const float max_dist = 100.f,
                                        const float surf_dist = 0.001f,
                                        const uint32_t max_step = 100) {
    float d0 = 0.;
    for (uint32_t step = 0; step < max_step; step++) {
        auto p = ro + d0 * rd;
        auto dist = get_dist(p, user_data);
        d0 += dist;
        if (d0 > max_dist || abs(dist) < surf_dist) {
            break;
        }
    }
    return d0;
}

UTILS_IN_CUDA inline Eigen::Vector3f get_ray_dir(const Eigen::Vector2f &uv, const Eigen::Vector3f &ro,
                                                 const Eigen::Vector3f &center,
                                                 const Eigen::Vector3f &up, const float screen_dist) {
    auto f = (center - ro).normalized();
    auto r = f.cross(up).normalized();
    auto u = r.cross(f);

    auto to_screen_center = f * screen_dist;
    auto to_screen_uv = to_screen_center + uv.x() * r + uv.y() * u;

    return to_screen_uv.normalized();
}

UTILS_IN_CUDA inline Eigen::Vector3f get_normal(const Eigen::Vector3f &p, get_dist_func get_dist,
                                                void *user_data = nullptr, const float ds = 0.0001f) {
    const Eigen::Vector2f e{ds, 0};
    const auto dist = get_dist(p, user_data);
    auto n =
            Eigen::Vector3f{dist, dist, dist} -
            Eigen::Vector3f{get_dist(p - Eigen::Vector3f{e.x(), e.y(), e.y()}, user_data),
                            get_dist(p - Eigen::Vector3f{e.y(), e.x(), e.y()}, user_data),
                            get_dist(p - Eigen::Vector3f{e.y(), e.y(), e.x()}, user_data)};

    return n.normalized();
}

UTILS_IN_CUDA inline float get_light(const Eigen::Vector3f &p, const Eigen::Vector3f &light_p,
                                     get_dist_func get_dist, void *user_data = nullptr, const float max_dist = 100.f,
                                     const float surf_dist = 0.001f, const uint32_t max_step = 100) {
    auto l = (light_p - p).normalized();
    auto n = get_normal(p, get_dist, user_data);

    float dif = std::clamp(n.dot(l), 0.f, 1.f);
    float d = ray_marching(p + n * surf_dist * 2., l, get_dist, user_data);
    if (d < (light_p - p).norm())
        dif *= .1;

    return dif;
}

// generic functions

template<typename T>
struct is_matrix : std::false_type {
};

template<typename Scalar_, int Rows_, int Cols_, int Options_, int MaxRows_, int MaxCols_>
struct is_matrix<Eigen::Matrix<Scalar_, Rows_, Cols_, Options_, MaxRows_, MaxCols_>> : std::true_type {
};

template<typename T>
struct is_vector : std::false_type {
};

template<typename Type, int Size>
struct is_vector<Eigen::Vector<Type, Size>> : std::true_type {
};

template<typename T>
UTILS_IN_CUDA auto smoothstep(const T &t1, const T &t2, const T &x) {
    if constexpr (is_vector<T>::value) {
        auto new_x_array = ((x - t1).array() / (t2 - t1).array()).min(1).max(0);
        return (new_x_array * new_x_array * (3 - 2 * new_x_array)).matrix();
    }
    else {
        auto new_x = std::clamp<T>((x - t1) / (t2 - t1), 0.0, 1.0);
        return new_x * new_x * (3 - 2 * new_x);
    }
}

template<typename T>
UTILS_IN_CUDA auto mix(const T &t1, const T &t2, float a) {
    return t1 * (1 - a) + t2 * a;
}

template<typename T>
UTILS_IN_CUDA float bond(const T &t1, const T &t1_pulse, const T &t2, const T &t2_pulse, const T &x) {
    if constexpr (is_vector<T>::value) {
        return (smoothstep(t1, t1_pulse, x).array() * smoothstep(t2, t2_pulse, x).array()).prod();
    }
    else {
        return smoothstep(t1, t1_pulse, x) * smoothstep(t2, t2_pulse, x);
    }
}

UTILS_IN_CUDA inline Eigen::Vector3f reflect(const Eigen::Vector3f &i, const Eigen::Vector3f &n) {
    return i - 2.0 * n * n.dot(i);
}


template<std::floating_point T>
UTILS_IN_CUDA auto template_min(const T a, const T b) {
    return std::min(a, b);
};

template<std::floating_point T, std::floating_point... Args>
    requires std::conjunction_v<std::is_same<T, Args>...>
UTILS_IN_CUDA auto template_min(const T a, const Args... b) {
    return std::min(a, template_min(b...));
};

template<std::floating_point T>
UTILS_IN_CUDA auto template_max(const T a, const T b) {
    return std::max(a, b);
};

template<std::floating_point T, std::floating_point... Args>
    requires std::conjunction_v<std::is_same<T, Args>...>
UTILS_IN_CUDA auto template_max(const T a, const Args... b) {
    return std::max(a, template_max(b...));
};

template<std::floating_point T, std::floating_point... Args>
    requires std::conjunction_v<std::is_same<T, Args>...>
UTILS_IN_CUDA auto template_sum(T first, Args... a) {
    return first + (a + ...);
};

template<std::floating_point T, std::floating_point... Args>
    requires std::conjunction_v<std::is_same<T, Args>...>
UTILS_IN_CUDA inline std::pair<T, T> min2(T first, Args... a) {
    T arr[] = {first, a...};
    T m1, m2; //存储两个最小值
    m1 = std::numeric_limits<T>::max();
    m2 = std::numeric_limits<T>::max();
    for (size_t i = 0; i < std::size(arr); i++) {
        if (arr[i] < m1) {
            m2 = m1;
            m1 = arr[i];
        }
        else if (arr[i] < m2) {
            m2 = arr[i];
        }
    }
    return std::pair<T, T>(m1, m2);
}


template<std::floating_point T>
UTILS_IN_CUDA inline float smin(float k, T a, T b) {
    return std::min(a, b) - pow(std::max(0.f, k - std::abs(a - b)), 2.f) / (4.f * k);
}

template<std::floating_point T>
UTILS_IN_CUDA inline float smin_low_high(float k, T low, T high) {
    return low - pow(std::max(0.f, k - (high - low)), 2.f) / (4.f * k);
}

template<std::floating_point... T>
UTILS_IN_CUDA inline float smin(float k, T... a) {
    auto [min_1,min_2] = min2(a...);
    return smin_low_high(k, min_1, min_2);
}

// 3d functions

UTILS_IN_CUDA inline Eigen::Vector3f spherical_coordinates(const float phi, const float theta,
                                                           const Eigen::Vector3f &to_z,
                                                           const Eigen::Vector3f &to_x) {
    auto z = to_z.normalized();
    auto y = z.cross(to_x).normalized();
    auto x = y.cross(z);

    const auto sin_theta = sin(theta);

    return x * std::cos(phi) * sin_theta + y * std::sin(phi) * sin_theta + z * std::cos(theta);
}


#undef UTILS_IN_CUDA

#endif //CPU_SHADER_UTILS_H
