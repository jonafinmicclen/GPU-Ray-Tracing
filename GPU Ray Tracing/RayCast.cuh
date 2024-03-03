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

__global__ void traceRay(Ray* rays, Triangle* triangles, int* thread_per_block, Vec3* point_lights)
{
	Vec3 intersection_point;
	int i = threadIdx.x + blockIdx.x * *thread_per_block;
	float shortest_distance = 99999999;
	Vec3 nearest_normal;
	Vec3 nearest_intersection_point;
	bool intersected = false;

	for (int triangle_index = 0; triangle_index < 24; ++triangle_index)
	{
		if (rays[i].intersectTriangle(triangles[triangle_index], intersection_point))
		{
			// Test if is nearest intersection
			float distance = (intersection_point.subtract(rays[i].origin)).length();
			if (distance < shortest_distance)
			{
				rays[i].color = triangles[triangle_index].color;
				intersected = true;
				shortest_distance = distance;
				nearest_normal = triangles[triangle_index].normal;
				nearest_intersection_point = intersection_point;
			}
		}
	}

	if (!intersected) {
		rays[i].color = { 0,0,0 };
		return;
	}

	// Occlusion testing, only one light source fixed rn
	Ray occlusion_ray;
	occlusion_ray.direction = (point_lights[0].subtract(nearest_intersection_point)).normalised();
	occlusion_ray.origin = nearest_intersection_point.add(occlusion_ray.direction.scalarMultiply(0.001));
	float distance_to_light = occlusion_ray.origin.subtract(point_lights[0]).length();
	for (int triangle_index = 0; triangle_index < 24; ++triangle_index)
	{
		if (occlusion_ray.intersectTriangle(triangles[triangle_index], intersection_point)  && (intersection_point.subtract(occlusion_ray.origin)).length() < distance_to_light)
		{
			rays[i].color = { 0,0,0 };
			return;
		}
	}
	rays[i].color = rays[i].color.scalarDivide((shortest_distance + (occlusion_ray.origin.subtract(point_lights[0])).length()) * 10);



	return;
}

Ray* CUDA_rays;
Triangle* CUDA_triangles;
int* CUDA_THREAD_PER_BLOCK;
Vec3* CUDA_point_lights;

void allocateMemory(Camera* camera) {

	size_t rays_size = camera->number_of_rays * sizeof(Ray);
	size_t triangles_size = camera->scene.triangles.size() * sizeof(Triangle);
	size_t lights_size = camera->scene.point_lights.size() * sizeof(Vec3);

	cudaMalloc(&CUDA_rays, rays_size);
	cudaMalloc(&CUDA_triangles, triangles_size);
	cudaMalloc(&CUDA_THREAD_PER_BLOCK, thread_per_block);
	cudaMalloc(&CUDA_point_lights, lights_size);

	cudaMemcpy(CUDA_rays, camera->rays_through_screen, rays_size, cudaMemcpyHostToDevice);
	cudaMemcpy(CUDA_triangles, camera->scene.triangles.data(), triangles_size, cudaMemcpyHostToDevice);
	cudaMemcpy(CUDA_THREAD_PER_BLOCK, &thread_per_block, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(CUDA_point_lights, camera->scene.point_lights.data(), sizeof(lights_size), cudaMemcpyHostToDevice);

}

void reCopyTris(Camera* camera) {
	cudaMemcpy(CUDA_triangles, camera->scene.triangles.data(), camera->scene.triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
}

void renderScreen(Camera* camera)
{

	// Will need to change this if change screen dimensions
	traceRay <<< block_size, thread_per_block >>> (CUDA_rays, CUDA_triangles, CUDA_THREAD_PER_BLOCK, CUDA_point_lights);

	cudaMemcpy(camera->rays_through_screen, CUDA_rays, camera->number_of_rays * sizeof(Ray), cudaMemcpyDeviceToHost);

}

void freeMemory() {
	cudaFree(CUDA_rays);
	cudaFree(CUDA_triangles);
	cudaFree(CUDA_THREAD_PER_BLOCK);
	cudaFree(CUDA_point_lights);
}