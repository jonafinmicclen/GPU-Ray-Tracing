#include "Main.cuh"

void renderScreen(Camera* camera)
{
	Ray* CUDA_rays = camera->rays_through_screen;
	Triangle* CUDA_triangles = camera->scene.triangles.data();
	int HOST_num_of_triangles = static_cast<int>(camera->scene.triangles.size());
	int* CUDA_num_of_triangles = &HOST_num_of_triangles;

	cudaMalloc(&CUDA_rays, sizeof(CUDA_rays));
	cudaMalloc(&CUDA_triangles, sizeof(CUDA_triangles));
	cudaMalloc(&CUDA_num_of_triangles, sizeof(CUDA_num_of_triangles));

	cudaMemcpy(CUDA_rays, camera->rays_through_screen, sizeof(CUDA_rays), cudaMemcpyHostToDevice);
	cudaMemcpy(CUDA_triangles, camera->scene.triangles.data(), sizeof(CUDA_triangles), cudaMemcpyHostToDevice);

	traceRay <<< 1, camera->number_of_rays >>> (CUDA_rays, CUDA_triangles);

	cudaMemcpy(camera->rays_through_screen, CUDA_rays, sizeof(CUDA_rays), cudaMemcpyDeviceToHost);
}

int main()
{
	int a[] = { 1,2,3 };
	int b[] = { 1,2,3 };
	int c[sizeof(a) / sizeof(int)] = { 0 };

	int* cudaA = 0;
	int* cudaB = 0;
	int* cudaC = 0;

	cudaMalloc(&cudaA, sizeof(a));
	cudaMalloc(&cudaB, sizeof(b));
	cudaMalloc(&cudaC, sizeof(c));

	cudaMemcpy(cudaA, a, sizeof(a), cudaMemcpyHostToDevice);
	cudaMemcpy(cudaB, b, sizeof(b), cudaMemcpyHostToDevice);

	vectorAdd <<< 1, sizeof(a) / sizeof(int) >>> (cudaA, cudaB, cudaC);

	cudaMemcpy(c, cudaC, sizeof(c), cudaMemcpyDeviceToHost);

	// Need triangle and ray
	Triangle h_triangle;
	h_triangle.vertecies[0] = { 0,0,0 };
	h_triangle.vertecies[1] = { 1,1,0 };
	h_triangle.vertecies[2] = { 2,0,0 };
	h_triangle.normal = { 0,0,-1 };

	Ray h_ray;
	h_ray.origin = { 0, 0, 1 };
	h_ray.direction = { 0,0,1 };

	Ray* d_ray;
	Triangle* d_triangle;

	cudaMalloc(&d_ray, sizeof(Ray));
	cudaMalloc(&d_triangle, sizeof(Triangle));

	cudaMemcpy(d_ray, &h_ray, sizeof(Ray), cudaMemcpyHostToDevice);
	cudaMemcpy(d_triangle, &h_triangle, sizeof(Triangle), cudaMemcpyHostToDevice);

	rayCastCUDA <<< 1, 1 >>> (d_ray, d_triangle);

	cudaMemcpy(&h_ray, d_ray, sizeof(Ray), cudaMemcpyDeviceToHost);

	std::cout << h_ray.collisionPoint.x;
	std::cout << h_ray.collisionPoint.y;
	std::cout << h_ray.collisionPoint.z;

	printf("%d\n", c[1]);

	// YOLOOO
	Camera* camera = new Camera;
	camera->scene.triangles = createTriCube(0.5f).triangles;
	camera->initialiseRaysThroughScreen();
	renderScreen(camera);

	return;

}