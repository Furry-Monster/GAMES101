//
// Created by LEI XU on 4/27/19.
//

#ifndef RASTERIZER_SHADER_H
#define RASTERIZER_SHADER_H
#include "Texture.hpp"
#include <eigen3/Eigen/Eigen>

struct fragment_shader_payload {
  fragment_shader_payload() { texture = nullptr; }

  fragment_shader_payload(const Eigen::Vector3f &col,
                          const Eigen::Vector3f &nor, const Eigen::Vector2f &tc,
                          Texture *tex)
      : color(col), normal(nor), tex_coords(tc), texture(tex) {}

  Eigen::Vector3f view_pos; // this prop is really disturbing,which should be
                            // named as shading_pos ,i think.
  Eigen::Vector3f color;
  Eigen::Vector3f normal;
  Eigen::Vector2f tex_coords;
  Texture *texture;
};

struct vertex_shader_payload {
  Eigen::Vector3f position;
};

#endif // RASTERIZER_SHADER_H
