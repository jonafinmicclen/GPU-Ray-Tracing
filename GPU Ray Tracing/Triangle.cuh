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

	__host__ void calculateNormal() {
		try {
			normal = (vertecies[1].subtract(vertecies[0])).cross(vertecies[2].subtract(vertecies[0])).normalised();
		}
		catch (std::exception& e) {
			std::cerr << "Error calculating normal: " << e.what() << std::endl;
		}
	}
};