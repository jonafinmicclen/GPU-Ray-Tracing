#pragma once

#include "Vec3.cuh"
#include "Triangle.cuh"

struct Ray {
	Vec3 origin;
	Vec3 direction;
	Vec3 collisionPoint;
	Vec3 color;
	bool has_collided = false;

	// returns the reflected ray
	__device__ Ray reflect(const Vec3& normal, const Vec3& intersectionPoint) {

		Ray newRay = *this;
		newRay.direction = direction.subtract(normal.normalised().scalarMultiply(2.0).scalarMultiply(direction.dot(normal.normalised())));
		newRay.origin = intersectionPoint;
		return newRay;
	}

	__device__ bool intersectTriangle(const Triangle& triangle, Vec3& intersectionPoint) const {

		// Check if the ray is parallel to the triangle
		double dotProduct = triangle.normal.dot(direction);
		if (std::abs(dotProduct) < 1e-20) {
			return false;
		}

		// Calculate the distance from the ray origin to the triangle plane
		double t = triangle.normal.dot(triangle.vertecies[0].subtract(origin)) / dotProduct;

		// Check if the intersection point is behind the ray origin
		if (t < 0) {
			return false;
		}

		// Calculate the intersection point
		intersectionPoint = origin.add(direction.scalarMultiply(t));

		// Check if the intersection point is inside the triangl
		Vec3 edge1 = triangle.vertecies[1].subtract(triangle.vertecies[0]);
		Vec3 edge2 = triangle.vertecies[2].subtract(triangle.vertecies[0]);
		Vec3 edge3 = intersectionPoint.subtract(triangle.vertecies[0]);

		double detT = edge1.cross(edge2).dot(triangle.normal);
		double u = edge3.cross(edge2).dot(triangle.normal) / detT;
		double v = edge1.cross(edge3).dot(triangle.normal) / detT;

		return (u >= 0 && v >= 0 && u + v <= 1);
	}
};