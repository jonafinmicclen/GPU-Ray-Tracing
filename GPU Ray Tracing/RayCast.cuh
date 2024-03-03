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

__global__ void traceRay(Ray* rays, Triangle* triangles)
{
	Vec3 intersection_point;
	int i = threadIdx.x;
	float shortest_distance = 1000000000000;
	bool intersected = false;

	for (int triangle_index = 0; triangle_index < 12; ++triangle_index)
	{
		if (rays[i].intersectTriangle(triangles[triangle_index], intersection_point))
		{
			// Test if is nearest intersection
			if ((intersection_point.subtract(rays[i].origin)).length() < shortest_distance)
			{
				// Does this make a copy of the triangle?
				rays[threadIdx.x].color = triangles[triangle_index].color;
				intersected = true;
			}
		}
	}

	if (!intersected) {
		rays[threadIdx.x].color = { 0,0,0 };
	}

	return;
}

void rayCastC() {
	return;
}