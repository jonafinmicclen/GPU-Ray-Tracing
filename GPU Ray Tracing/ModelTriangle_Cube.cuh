#pragma once

#include "ModelTriangle.cuh"
#include "ModelStandard.cuh"
#include "Vec3.cuh"
#include <tuple>
#include <vector>

ModelStandard createCube(double sideLength) {
    // Define the vertices of the cube
    Vec3 v1 = { -sideLength / 2, -sideLength / 2, -sideLength / 2 };
    Vec3 v2 = { sideLength / 2, -sideLength / 2, -sideLength / 2 };
    Vec3 v3 = { sideLength / 2, sideLength / 2, -sideLength / 2 };
    Vec3 v4 = { -sideLength / 2, sideLength / 2, -sideLength / 2 };
    Vec3 v5 = { -sideLength / 2, -sideLength / 2, sideLength / 2 };
    Vec3 v6 = { sideLength / 2, -sideLength / 2, sideLength / 2 };
    Vec3 v7 = { sideLength / 2, sideLength / 2, sideLength / 2 };
    Vec3 v8 = { -sideLength / 2, sideLength / 2, sideLength / 2 };

    // Define the surfaces of the cube using vertex indices
    std::vector<std::tuple<int, int, int>> surfaces = {
        {0, 1, 2}, {0,3,2},
        {4, 5, 6}, {4,7,6},
        {0, 4, 7}, {7,0,3},
        {1, 5, 6}, {6, 2, 1},
        {0, 1, 5}, {5, 4, 0},
        {2, 3, 7}, {7, 6, 2}
    };
    std::vector<Vec3> colors = {
        {1,0,0}, {1,0,0},
        {0,1,0}, {0,1,0},
        {0,0,1}, {0,0,1},
        {0.5,0.5,0}, {0.5,0.5,0},
        {0,0.5,0.5}, {0,0.5,0.5},
        {0.5,0,0.5}, {0.5,0,0.5}
    };

    // Create the cube instance of standardModel
    return { {v1, v2, v3, v4, v5, v6, v7, v8}, surfaces, colors };
}

ModelTriangle createTriCube(float sideLength) {
    ModelStandard cube_std = createCube(sideLength);
    return cube_std.convertToTriModel();
}