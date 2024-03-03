#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

struct Vec3 
{
	float x, y, z;

    __host__ __device__ Vec3 normalised() const {
        return scalarDivide(length());
    }

    __host__ __device__ float length() const {
        return sqrtf(x * x + y * y + z * z);
    }

    // Device method to add two vectors
    __host__ __device__ Vec3 add(const Vec3& other) const {
        return { x + other.x, y + other.y, z + other.z };
    }

    // Device method to subtract two vectors
    __host__ __device__ Vec3 subtract(const Vec3& other) const {
        return { x - other.x, y - other.y, z - other.z };
    }

    // Returns the cross product
    __host__ __device__ Vec3 cross(const Vec3& other) const {
        return Vec3({
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
            });
    }

    __host__ __device__ Vec3 scalarDivide(const float& scalar) const {
        return { x / scalar, y / scalar, z / scalar };
    }

    __host__ __device__ Vec3 scalarMultiply(const float& scalar) const {
        return { x * scalar, y * scalar, z * scalar };
    }

    // Device method to calculate the dot product of two vectors
    __host__ __device__ float dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
};