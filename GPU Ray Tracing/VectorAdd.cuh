#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void vectorAdd(int* a, int* b, int* c)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];

	return;
}