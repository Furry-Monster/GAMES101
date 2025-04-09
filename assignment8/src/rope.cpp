#include <iostream>
#include <vector>

#include "CGL/vector2D.h"

#include "mass.h"
#include "rope.h"
#include "spring.h"

namespace CGL {

Rope::Rope(Vector2D start, Vector2D end, int num_nodes, float node_mass,
           float k, vector<int> pinned_nodes) {
  // TODO (Part 1): Create a rope starting at `start`, ending at `end`, and
  // containing `num_nodes` nodes. node_mass: 质点质量; k: 弹簧系数
  Vector2D delta = (end - start) / (num_nodes - 1);
  // 质点
  for (int i = 0; i < num_nodes; i++) {
    Vector2D position = start + i * delta;
    // 终点位置
    Mass *e = new Mass(position, node_mass, false);
    // 弹簧
    if (i > 0) {
      // 起始位置
      Mass *s = masses.back();
      Spring *spring = new Spring(s, e, k);
      springs.push_back(spring);
    }
    masses.push_back(e);
  }
  for (auto &i : pinned_nodes) {
    masses[i]->pinned = true;
  }
}

void Rope::simulateEuler(float delta_t, Vector2D gravity) {
  for (auto &s : springs) {
    // TODO (Part 2): Use Hooke's law to calculate the force on a node
    float l1 = s->rest_length;
    // a: m1; b: m2
    Vector2D ab = ((s->m2)->position - (s->m1)->position);
    float l2 = ab.norm();
    // a收到的力
    Vector2D f = s->k * ab / l2 * (l2 - l1);
    (s->m1)->forces += f;
    (s->m2)->forces += -f;
  }
  // int cnt = 0;
  for (auto &m : masses) {
    if (!m->pinned) {
      // TODO (Part 2): Add the force due to gravity, then compute the new
      // velocity and position 考虑重力
      m->forces += gravity * m->mass;
      // TODO (Part 2): Add global damping
      m->forces -= kd_euler * m->velocity;
      Vector2D a = m->forces / m->mass;
      // 显式欧拉
      // m->position += m->velocity * delta_t;
      // m->velocity += a * delta_t;
      // 隐式欧拉
      m->velocity += a * delta_t;
      m->position += m->velocity * delta_t;
    }

    // Reset all forces on each mass
    m->forces = Vector2D(0, 0);
  }
}

void Rope::simulateVerlet(float delta_t, Vector2D gravity) {
  for (auto &s : springs) {
    // TODO (Part 3): Simulate one timestep of the rope using explicit Verlet
    // （solving constraints)
    float l1 = s->rest_length;
    // a: m1; b: m2
    Vector2D ab = ((s->m2)->position - (s->m1)->position);
    float l2 = ab.norm();
    // a收到的力
    Vector2D f = s->k * ab / l2 * (l2 - l1);
    (s->m1)->forces += f;
    (s->m2)->forces += -f;
  }

  for (auto &m : masses) {
    if (!m->pinned) {
      Vector2D temp_position = m->position;
      // TODO (Part 3.1): Set the new position of the rope mass
      // 考虑重力
      m->forces += gravity * m->mass;
      Vector2D a = m->forces / m->mass;
      // TODO (Part 4): Add global Verlet damping
      Vector2D pos = m->position +
                     (1 - kd_verlet) * (m->position - m->last_position) +
                     a * delta_t * delta_t;
      m->last_position = m->position;
      m->position = pos;
    }
    // Reset all forces on each mass
    m->forces = Vector2D(0, 0);
  }
}
} // namespace CGL
