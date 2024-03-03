#include "Camera.cuh"

int Camera::indexFromCoordinate(const int x, const int y)
{
	return y * width + x;
}

void Camera::initialiseRaysThroughScreen()
{
	int index = 0;
	for (int x = 0; x < width; ++x) {
		for (int y = 0; y < height; ++y) {
			++index;

			Vec3 center_of_screen = origin.add(forward_vector.scalarMultiply(distance_to_screen));
			Vec3 point_on_screen = center_of_screen.add(up_vector.scalarMultiply(pixel_size).scalarMultiply(y - height / 2)).add(side_vector.scalarMultiply(pixel_size).scalarMultiply(x - width / 2));

			rays_through_screen[index].origin = origin;
			rays_through_screen[index].direction = point_on_screen.subtract(origin).normalised();
		}
	}
}
