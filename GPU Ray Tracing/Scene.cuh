#pragma once

#include "Triangle.cuh"
#include "vector"

struct Scene {
	std::vector<Triangle> triangles;
	std::vector<Vec3> point_lights;
};