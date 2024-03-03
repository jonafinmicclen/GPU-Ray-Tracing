#pragma once

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// OpenGL
#include <GL/glut.h>

// Standard
#include <stdio.h>

// Program specific
#include "Vec3.cuh"
#include "Triangle.cuh"
#include "RayCast.cuh"
#include "Camera.cuh"

// Tests
#include "ModelTriangle_Cube.cuh"
#include "ModelPermutations.cuh"