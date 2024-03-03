#include "Main.cuh"

Camera* camera = new Camera;

void init() {
    glClearColor(1.0, 1.0, 1.0, 1.0); // Set clear color to white
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, camera->width, 0.0, camera->height); // Set the coordinate system
}

int main(int argc, char** argv)
{
    
	camera->scene.triangles = createTriCube(5.0f).triangles;
    camera->scene.point_lights.push_back({ -8,10,0 });
	camera->initialiseRaysThroughScreen();

	// Allocate memory on GPU
	allocateMemory(camera);
    renderScreen(camera);

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(camera->width, camera->height);
    glutCreateWindow("CUDA Ray tracing");

    init();

    Vec3 rot_vec = { 2,1,0 };
    rot_vec = rot_vec.normalised();

    while (1) {
        glClear(GL_COLOR_BUFFER_BIT);

        glBegin(GL_POINTS);

        // Permutate object
        for (auto& triangle : camera->scene.triangles) {
            for (auto& vertex : triangle.vertecies) {
                rotateVec3(vertex, 0.1, rot_vec);
            }
            triangle.calculateNormal();
        }


        // Handle rendering
        reCopyTris(camera);
        renderScreen(camera);
        int index = 0;
        for (int x = 0; x < camera->width; ++x) {
            for (int y = 0; y < camera->height; ++y) {
                Ray* currentRay = &camera->rays_through_screen[index];
                glColor3f(camera->rays_through_screen[index].color.x * 255, camera->rays_through_screen[index].color.y * 255, camera->rays_through_screen[index].color.z * 255);
                glVertex2i(x, y);

                ++index;
            }
        }

        glEnd();

        glFlush();
    }

	// Free memory on GPU
	freeMemory();


	return 0;
}