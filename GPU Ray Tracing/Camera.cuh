#pragma once

#include <utility>
#include <tuple>

#include "Vec3.cuh"
#include "Ray.cuh"
#include "Scene.cuh"

class Camera {
public:

	// Perspective configuration
	static const int width = 1920;
	static const int height = 1080;
	const float screen_scene_ratio = 0.01f;
	static const int distance_to_screen = 5;
	const Vec3 origin_vector = { -8.0f,0,0 };
	const Vec3 look_at_vector = { 1,0,0 };

	// Scene info
	static const int number_of_rays = width * height;
	Scene scene;
	Ray rays_through_screen[number_of_rays];
	
	// Render buffer
	Vec3 screen_buffer[number_of_rays];

	// Utility
	int indexFromCoordinate(const int x, const int y);

	// Rendering
	void initialiseRaysThroughScreen();

	// Math
	__device__ __host__ std::pair<Vec3,Vec3> getSideUpVectors();
};