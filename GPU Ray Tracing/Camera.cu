#include "Camera.cuh"

int Camera::indexFromCoordinate(const int x, const int y)
{
	return y * width + x;
}

void Camera::initialiseRaysThroughScreen()
{
	Vec3 up_vector, side_vector;
	std::tie(side_vector, up_vector) = getSideUpVectors();

	int index = 0;
	for (int x = 0; x < width; ++x) {
		for (int y = 0; y < height; ++y) {

			Vec3 center_of_screen = origin_vector.add(look_at_vector.scalarMultiply(distance_to_screen));
			Vec3 top_center_of_screen = center_of_screen.add(up_vector.scalarMultiply((y - height/2) * screen_scene_ratio ));
			Vec3 point_on_screen = top_center_of_screen.add(side_vector.scalarMultiply((x - width/2) * screen_scene_ratio ));

			rays_through_screen[index].origin = origin_vector;
			rays_through_screen[index].color = { 0,0,0 };
			rays_through_screen[index].direction = (point_on_screen.subtract(origin_vector)).normalised();

			++index;
		}
	}
}

__device__ __host__ std::pair<Vec3, Vec3> Camera::getSideUpVectors()
{
	if (look_at_vector.x != 0) {
		Vec3 vec1 = Vec3({-look_at_vector.y, look_at_vector.x, 0.0f}).normalised();
		return std::make_pair(look_at_vector.normalised().cross(vec1), vec1);
	}
	else {
		Vec3 vec1 = Vec3({ 0.0f, -look_at_vector.z, look_at_vector.y }).normalised();
		return std::make_pair(look_at_vector.normalised().cross(vec1), vec1);
	}
}