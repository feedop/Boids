﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "boidsGPU.cuh"
#include "openGLsetup.hpp"
#include "defines.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "cudaGL.h"
#include "cuda_gl_interop.h"
#include <stdio.h>
#include <iostream>
#include <sstream>

// NVIDIA GPU selector for devices with multiple GPUs (e.g. laptops)
extern "C"
{
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}

void handleInput(int argc, char* argv[], int& boidCount, bool& calculateOnCPU);

template<bool calculateOnCPU>
void eventLoop(GLFWwindow* window, const int boidCount);

void initializeBuffer(GLuint* VAO, GLuint* VBO, const int boidCount);

void usage()
{
    std::cout << "USAGE: ./Boids [boidCount] [-c]\n";
    exit(0);
}

int main(int argc, char* argv[])
{
    int boidCount;
    bool calculateOnCPU = false;
    cudaError_t cudaStatus;

    handleInput(argc, argv, boidCount, calculateOnCPU);
    
    cudaStatus = cudaGLSetGLDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    GLFWwindow* window;

    // Initialize the library
    if (!glfwInit())
        return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(WIDTH, HEIGHT, "Boids", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Load GL and set the viewport to match window size
    gladLoadGL();
    std::cout << glGetString(GL_VERSION) << std::endl;
    glViewport(0, 0, WIDTH, HEIGHT);

    // Main event loop
    if (calculateOnCPU)
    {
        eventLoop<true>(window, boidCount);
    }
    else
    {
        eventLoop<false>(window, boidCount);
    }

    // Cleanup
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

void handleInput(int argc, char* argv[], int& boidCount, bool& calculateOnCPU)
{
    if (argc == 2)
    {
        if (!strcmp(argv[1], "-c"))
        {
            calculateOnCPU = true;
            boidCount = DEFAULT_BOIDCOUNT;
        }
        else
        {
            boidCount = atoi(argv[1]);
            if (boidCount <= 0)
            {
                usage();
            }
        }  
    }
    else if (argc == 3)
    {
        if (!strcmp(argv[1], "-c"))
        {
            calculateOnCPU = true;
            boidCount = atoi(argv[2]);
            if (boidCount <= 0)
            {
                usage();
            }
        }
        else
        {
            boidCount = atoi(argv[1]);
            if (boidCount <= 0)
            {
                usage();
            }
            if (!strcmp(argv[2], "-c"))
            {
                calculateOnCPU = true;
            }
            else
            {
                usage();
            }
        }
    }
    else if (argc > 3)
    {
        usage();
    }
    else
    {
        boidCount = DEFAULT_BOIDCOUNT;
    }
}

template<bool calculateOnCPU>
void eventLoop(GLFWwindow* window, const int boidCount)
{
    // Set up Shaders
    GLuint vertexShader, fragmentShader, shaderProgram;
    initializeShaders(vertexShader, fragmentShader, shaderProgram);

    // Initialize vertex buffer
    GLuint VAO, VBO;
    
    initializeBuffer(&VAO, &VBO, boidCount);

    // Boid positions
    float* boidX = 0;
    float* boidY = 0;
    // Boid velocity vectors
    float* boidDX = 0;
    float* boidDY = 0;

    // Fill the initial array with random values
    if constexpr (calculateOnCPU)
    {

    }
    else
    {
        cudaGLRegisterBufferObject(VBO);
        GPU::initializeBoidLists(&boidX, &boidY, &boidDX, &boidDY, boidCount);
        GPU::generateRandomPositions(VBO, boidX, boidY, boidDX, boidDY, boidCount);
    }    

    double lastTime = glfwGetTime();
    int frameCount = 0;
    int seconds = 0;

    // Loop until the user closes the window 
    while (!glfwWindowShouldClose(window))
    {
        // Clear 
        glClearColor(0.08f, 0.17f, 0.43f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Calculate new positions of boids in each frame

        if constexpr (calculateOnCPU)
        {

        }
        else
        {
            GPU::calculatePositions(VBO, boidX, boidY, boidDX, boidDY, boidCount);
        }

        // Render triangles
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);

        glDrawArrays(GL_TRIANGLES, 0, boidCount * 3);

        // Swap front and back buffers 
        glfwSwapBuffers(window);

        // Show FPS
        double currentTime = glfwGetTime();
        double delta = currentTime - lastTime;
        if (delta >= 1.0)
        {
            double fps = double(frameCount) / delta;
            std::stringstream ss;
            ss << "Boids" << " " << " [" << fps << " FPS]";

            glfwSetWindowTitle(window, ss.str().c_str());
            lastTime = currentTime;
            frameCount = 0;
            /*seconds++;
            if (seconds >= 5)
                exit(0);*/
        }
        else
        {
            frameCount++;
        }

        // Poll for and process events
        glfwPollEvents();
    }

    // Cleanup

    if constexpr (calculateOnCPU)
    {

    }
    else
    {
        cudaGLUnregisterBufferObject(VBO);
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
}


