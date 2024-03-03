#pragma once

#include <vector>
#include <tuple>
#include "ModelTriangle.cuh"
#include "Vec3.cuh"
#include "Triangle.cuh"

class ModelStandard {
public:

	// Vertex coordinate for each vertex
	std::vector<Vec3> vertices;
	// Vertex index of each surface
	std::vector<std::tuple<int, int, int>> surfaces;
	// Material of each surface, with same indexing
	std::vector<Vec3> colors;

	// return triangularModel type version of this
	ModelTriangle convertToTriModel() const {

		// Initalises the model
		ModelTriangle triModel;
		int i = 0;

		for (auto& surface : surfaces) {

			Triangle tri;

			// Create tri object for poly in the model
			tri.vertecies[0] = vertices[std::get<0>(surface)];
			tri.vertecies[1] = vertices[std::get<1>(surface)];
			tri.vertecies[2] = vertices[std::get<2>(surface)];
			tri.calculateNormal();
			tri.color = colors[i];

			// Add the tri object to the new model
			triModel.triangles.push_back(tri);
			++i;
		}

		return triModel;
	}
};