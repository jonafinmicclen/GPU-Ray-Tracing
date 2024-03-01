#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Scene.cuh"
#include "Ray.cuh"

__global__ void rayCastCUDA(Ray* rays, Triangle* triangle)
{
	int i = threadIdx.x;
	Vec3 intersectionPoint;

	if (rays[i].intersectTriangle(*triangle, intersectionPoint)) 
	{
		rays[i].collisionPoint = intersectionPoint;
	}

	return;
}

void rayCastC() {
	return;
}