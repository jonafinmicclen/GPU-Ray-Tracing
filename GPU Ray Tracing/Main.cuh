#pragma once

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Standard
#include <stdio.h>

// Program specific
#include "VectorAdd.cuh"
#include "Vec3.cuh"
#include "Triangle.cuh"
#include "RayCast.cuh"
#include "Camera.cuh"

// Tests
#include "ModelTriangle_Cube.cuh"
#include "ModelPermutations.cuh"

#include <GL/glut.h>