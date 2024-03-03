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

			Vec3 center_of_screen = origin.add(forward_vector.scalarMultiply(distance_to_screen));
			Vec3 top_center_of_screen = center_of_screen.add(up_vector.scalarMultiply((y - height/2) * screen_scene_ratio ));
			Vec3 point_on_screen = top_center_of_screen.add(side_vector.scalarMultiply((x - width/2) * screen_scene_ratio ));

			rays_through_screen[index].origin = origin;
			rays_through_screen[index].color = { 0,0,0 };
			rays_through_screen[index].direction = (point_on_screen.subtract(origin)).normalised();

			++index;
		}
	}
}
