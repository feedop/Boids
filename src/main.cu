
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <iostream>

#define DEFAULT_BOIDCOUNT 2

void handleInput(int argc, char* argv[], int& boidCount);

void usage()
{
    std::cout << "USAGE: ./Boids [boidCount]\n";
    exit(0);
}

int main(int argc, char* argv[])
{
    int boidCount;
    cudaError_t cudaStatus;

    handleInput(argc, argv, boidCount);
    
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(800, 800, "Boids", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    // Load GL and set the viewport to match window size
    gladLoadGL();
    glViewport(0, 0, 800, 800);

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClearColor(0.08f, 0.17f, 0.43f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    // cleanup
    glfwTerminate();

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

void handleInput(int argc, char* argv[], int& boidCount)
{
    if (argc == 2)
    {
        boidCount = atoi(argv[1]);
        if (boidCount <= 0)
        {
            usage();
        }
    }
    else if (argc > 2)
    {
        usage();
    }
    else
    {
        boidCount = DEFAULT_BOIDCOUNT;
    }
}