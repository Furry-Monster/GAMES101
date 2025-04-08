#include "rasterizer.hpp"
#include <algorithm>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <vector>

rst::pos_buf_id
rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions) {
  auto id = get_next_id();
  pos_buf.emplace(id, positions);

  return {id};
}

rst::ind_buf_id
rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices) {
  auto id = get_next_id();
  ind_buf.emplace(id, indices);

  return {id};
}

rst::col_buf_id
rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols) {
  auto id = get_next_id();
  col_buf.emplace(id, cols);

  return {id};
}

auto to_vec4(const Eigen::Vector3f &v3, float w = 1.0f) {
  return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

// NOTE:this method will be used in
// void rst::rasterizer::rasterize_triangle(const Triangle &t)
// which is after void draw(),so the Vector3f represent the point after MVP
// mapping. and _v[3] is a represent of depth for z-buffer.
static bool insideTriangle(int x, int y, const Vector3f *_v) {
  Vector3f edges[3];
  Vector3f conns[3];
  for (int i = 0; i < 3; i++) {
    edges[i] = _v[(i + 1) % 3] - _v[i];
    conns[i] = Vector3f(x - _v[i].x(), y - _v[i].y(), -_v[i].z());
  }
  Vector3f res[3];
  for (int i = 0; i < 3; i++) {
    res[i] = conns[i].cross(edges[i]);
  }
  return res[0].dot(res[1]) >= 0 && res[1].dot(res[2]) >= 0 &&
         res[2].dot(res[0]) >= 0;
}

static bool insideTriangle(float x, float y, const Vector3f *_v) {
  Vector3f edges[3];
  Vector3f conns[3];
  for (int i = 0; i < 3; i++) {
    edges[i] = _v[(i + 1) % 3] - _v[i];
    conns[i] = Vector3f(x - _v[i].x(), y - _v[i].y(), -_v[i].z());
  }
  Vector3f res[3];
  for (int i = 0; i < 3; i++) {
    res[i] = conns[i].cross(edges[i]);
  }
  return res[0].dot(res[1]) >= 0 && res[1].dot(res[2]) >= 0 &&
         res[2].dot(res[0]) >= 0;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y,
                                                            const Vector3f *v) {
  float c1 =
      (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y +
       v[1].x() * v[2].y() - v[2].x() * v[1].y()) /
      (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() +
       v[1].x() * v[2].y() - v[2].x() * v[1].y());
  float c2 =
      (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y +
       v[2].x() * v[0].y() - v[0].x() * v[2].y()) /
      (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() +
       v[2].x() * v[0].y() - v[0].x() * v[2].y());
  float c3 =
      (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y +
       v[0].x() * v[1].y() - v[1].x() * v[0].y()) /
      (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() +
       v[0].x() * v[1].y() - v[1].x() * v[0].y());
  return {c1, c2, c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer,
                           col_buf_id col_buffer, Primitive type) {
  auto &buf = pos_buf[pos_buffer.pos_id];
  auto &ind = ind_buf[ind_buffer.ind_id];
  auto &col = col_buf[col_buffer.col_id];

  float f1 = (50 - 0.1) / 2.0;
  float f2 = (50 + 0.1) / 2.0;

  Eigen::Matrix4f mvp = projection * view * model;
  for (auto &i : ind) {
    Triangle t;
    Eigen::Vector4f v[] = {mvp * to_vec4(buf[i[0]], 1.0f),
                           mvp * to_vec4(buf[i[1]], 1.0f),
                           mvp * to_vec4(buf[i[2]], 1.0f)};
    // Homogeneous division
    for (auto &vec : v) {
      vec /= vec.w();
    }
    // Viewport transformation
    for (auto &vert : v) {
      vert.x() = 0.5 * width * (vert.x() + 1.0);
      vert.y() = 0.5 * height * (vert.y() + 1.0);
      vert.z() = vert.z() * f1 + f2;
    }

    for (int i = 0; i < 3; ++i) {
      t.setVertex(i, v[i].head<3>());
      t.setVertex(i, v[i].head<3>());
      t.setVertex(i, v[i].head<3>());
    }

    auto col_x = col[i[0]];
    auto col_y = col[i[1]];
    auto col_z = col[i[2]];

    t.setColor(0, col_x[0], col_x[1], col_x[2]);
    t.setColor(1, col_y[0], col_y[1], col_y[2]);
    t.setColor(2, col_z[0], col_z[1], col_z[2]);

    rasterize_triangle(t);
  }
}

// Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle &t) {
  auto v = t.toVector4();
  bool MSAA = true;

  // Find out the bounding box of current triangle.
  // iterate through the pixel and find if the current pixel is inside the
  // triangle
  int max_x =
      static_cast<int>(std::max(std::max(v[0].x(), v[1].x()), v[2].x()));
  int max_y =
      static_cast<int>(std::max(std::max(v[0].y(), v[1].y()), v[2].y()));
  int min_x =
      static_cast<int>(std::min(std::min(v[0].x(), v[1].x()), v[2].x()));
  int min_y =
      static_cast<int>(std::min(std::min(v[0].y(), v[1].y()), v[2].y()));

  // If so, use the following code to get the interpolated z value.
  // auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
  // float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma /
  // v[2].w()); float z_interpolated = alpha * v[0].z() / v[0].w() + beta *
  // v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w(); z_interpolated *=
  // w_reciprocal;

  //  set the current pixel (use the set_pixel function) to the color of
  // the triangle (use getColor function) if it should be painted.
  if (!MSAA) {
    for (int ind_x = min_x; ind_x < max_x; ind_x++) {
      for (int ind_y = min_y; ind_y < max_y; ind_y++) {
        if (!insideTriangle(ind_x, ind_y, t.v))
          continue;
        auto [alpha, beta, gamma] = computeBarycentric2D(ind_x, ind_y, t.v);
        float w_reciprocal =
            1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
        float z_interpolated = alpha * v[0].z() / v[0].w() +
                               beta * v[1].z() / v[1].w() +
                               gamma * v[2].z() / v[2].w();
        z_interpolated *= w_reciprocal;

        int index = get_index(ind_x, ind_y);
        if (depth_buf[index] > z_interpolated) {
          depth_buf[index] = z_interpolated;
          set_pixel({ind_x, ind_y, z_interpolated}, t.getColor());
        }
      }
    }
  } else {
    // apply MSAA
    // conv core
    std::vector<Eigen::Vector2f> conv{
        {-0.25, -0.25},
        {0.25, -0.25},
        {0.25, -0.25},
        {0.25, 0.25},
    };
    for (int i = min_x; i < max_x; i++) {
      for (int j = min_y; j < max_y; j++) {
        int sample_timer = 0;
        float min_depth = FLT_MAX;
        for (int k = 0; k < 4; k++) {
          if (insideTriangle(i + conv[i][0], j + conv[i][1], t.v)) {
            auto [alpha, beta, gamma] =
                computeBarycentric2D(i + conv[i][0], j + conv[i][1], t.v);
            float w_reciprocal =
                1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
            float z_interpolated = alpha * v[0].z() / v[0].w() +
                                   beta * v[1].z() / v[1].w() +
                                   gamma * v[2].z() / v[2].w();
            z_interpolated *= w_reciprocal;
            min_depth = std::min(min_depth, z_interpolated);
            ++sample_timer;
          }
        }
        if (sample_timer != 0) {
          if (depth_buf[get_index(i, j)] > min_depth) {
            depth_buf[get_index(i, j)] = min_depth;
            set_pixel({i, j, min_depth}, t.getColor() * sample_timer / 4.0f);
          }
        }
      }
    }
  }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f &m) { model = m; }

void rst::rasterizer::set_view(const Eigen::Matrix4f &v) { view = v; }

void rst::rasterizer::set_projection(const Eigen::Matrix4f &p) {
  projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff) {
  if ((buff & rst::Buffers::Color) == rst::Buffers::Color) {
    std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
  }
  if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth) {
    std::fill(depth_buf.begin(), depth_buf.end(),
              std::numeric_limits<float>::infinity());
  }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h) {
  frame_buf.resize(w * h);
  depth_buf.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y) {
  return (height - 1 - y) * width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f &point,
                                const Eigen::Vector3f &color) {
  // old index: auto ind = point.y() + point.x() * width;
  auto ind = (height - 1 - point.y()) * width + point.x();
  frame_buf[ind] = color;
}

// clang-format on
