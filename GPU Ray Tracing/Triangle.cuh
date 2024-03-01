#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vec3.cuh"
#include "iostream"

struct Triangle
{
	Vec3 vertecies[3];
	Vec3 normal;
	Vec3 color;

};