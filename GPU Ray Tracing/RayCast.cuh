#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Scene.cuh"
#include "Ray.cuh"
#include "Camera.cuh"

int block_size = 4148;
int thread_per_block = 500;

__global__ void rayCastCUDA(Ray* rays, Triangle* triangles)
{
	int i = threadIdx.x + blockIdx.x;
	Vec3 intersectionPoint;

	for (int triangle_index = 0; triangle_index < 12; ++triangle_index)
	{
		if (rays[i].intersectTriangle(triangles[triangle_index], intersectionPoint))
		{
			rays[i].color = { 1,1,1 };
		}
	}

	return;
}

__global__ void traceRay(Ray* rays, Triangle* triangles, int* thread_per_block)
{
	Vec3 intersection_point;
	int i = threadIdx.x + blockIdx.x * *thread_per_block;
	float shortest_distance = 99999999;
	bool intersected = false;

	for (int triangle_index = 0; triangle_index < 12; ++triangle_index)
	{
		if (rays[i].intersectTriangle(triangles[triangle_index], intersection_point))
		{
			// Test if is nearest intersection
			float distance = (intersection_point.subtract(rays[i].origin)).length();
			if (distance < shortest_distance)
			{
				rays[i].color = triangles[triangle_index].color.scalarDivide(distance* 100);
				intersected = true;
				shortest_distance = distance;
			}
		}
	}

	if (!intersected) {
		rays[i].color = { 0,0,0 };
	}

	return;
}

Ray* CUDA_rays;
Triangle* CUDA_triangles;
int* CUDA_THREAD_PER_BLOCK;

void allocateMemory(Camera* camera) {

	size_t rays_size = camera->number_of_rays * sizeof(Ray);
	size_t triangles_size = camera->scene.triangles.size() * sizeof(Triangle);

	cudaMalloc(&CUDA_rays, rays_size);
	cudaMalloc(&CUDA_triangles, triangles_size);
	cudaMalloc(&CUDA_THREAD_PER_BLOCK, thread_per_block);

	cudaMemcpy(CUDA_rays, camera->rays_through_screen, rays_size, cudaMemcpyHostToDevice);
	cudaMemcpy(CUDA_triangles, camera->scene.triangles.data(), triangles_size, cudaMemcpyHostToDevice);
	cudaMemcpy(CUDA_THREAD_PER_BLOCK, &thread_per_block, sizeof(int), cudaMemcpyHostToDevice);
}

void reCopyTris(Camera* camera) {
	cudaMemcpy(CUDA_triangles, camera->scene.triangles.data(), camera->scene.triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
}

void renderScreen(Camera* camera)
{

	// Will need to change this if change screen dimensions
	traceRay <<< block_size, thread_per_block >>> (CUDA_rays, CUDA_triangles, CUDA_THREAD_PER_BLOCK);

	cudaMemcpy(camera->rays_through_screen, CUDA_rays, camera->number_of_rays * sizeof(Ray), cudaMemcpyDeviceToHost);

}

void freeMemory() {
	cudaFree(CUDA_rays);
	cudaFree(CUDA_triangles);
	cudaFree(CUDA_THREAD_PER_BLOCK);
}