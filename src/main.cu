#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "boidsGPU.cuh"
#include "boidsCPU.hpp"
#include "openGLsetup.hpp"
#include "parameterManager.hpp"
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

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

void printParameters(const ParameterManager* parameterManager);

void Clear();


void usage()
{
    std::cout << "USAGE: ./Boids [-c] [boidCount]\n";
    exit(0);
}

int main(int argc, char* argv[])
{
    int boidCount;
    bool calculateOnCPU = true;
    cudaError_t cudaStatus;

    handleInput(argc, argv, boidCount, calculateOnCPU);
    
    cudaStatus = cudaSetDevice(0);
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

// Handle input arguments
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

// OpenGL initialization and main loop
template<bool calculateOnCPU>
void eventLoop(GLFWwindow* window, const int boidCount)
{
    cudaGraphicsResource_t resource;
    float* cpuVBO;

    // Set up Shaders
    GLuint vertexShader, fragmentShader, shaderProgram;
    initializeShaders(vertexShader, fragmentShader, shaderProgram);

    // Initialize vertex buffer
    GLuint VAO, VBO;


    // Boid positions
    float* boidX = 0;
    float* boidY = 0;
    // Boid velocity vectors
    float* boidDX = 0;
    float* boidDY = 0;

    // Fill the initial array with random values
    if constexpr (calculateOnCPU)
    {
        initializeBuffer(&VAO, &VBO, boidCount);

        CPU::initializeBoidLists(&cpuVBO, &boidX, &boidY, &boidDX, &boidDY, boidCount);
        CPU::generateRandomPositions(boidX, boidY, boidDX, boidDY, boidCount);  
    }
    else
    {
        initializeBuffer(&VAO, &VBO, boidCount);

        GPU::initializeBoidLists(&boidX, &boidY, &boidDX, &boidDY, boidCount);
        GPU::generateRandomPositions(boidX, boidY, boidDX, boidDY, boidCount);

        HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, VBO, cudaGraphicsRegisterFlagsNone));
    }    

    double lastTime = glfwGetTime();
    int frameCount = 0;
    int seconds = 0;

    ParameterManager parameterManager(boidCount);

    // Set key callbacks
    glfwSetWindowUserPointer(window, static_cast<void*>(&parameterManager));
    glfwSetKeyCallback(window, keyCallback);

    // Print initial parameters
    printParameters(&parameterManager);

    // Loop until the user closes the window 
    while (!glfwWindowShouldClose(window))
    {
        // Clear 
        glClearColor(0.08f, 0.17f, 0.43f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Bind triangle vertices
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        // Calculate new positions of boids in each frame using CUDA
        if constexpr (calculateOnCPU)
        {
            CPU::calculatePositions(cpuVBO, boidX, boidY, boidDX, boidDY, boidCount, parameterManager);
        }
        else
        {
            GPU::calculatePositions(&resource, boidX, boidY, boidDX, boidDY, boidCount, parameterManager);
        }

        // Render triangles
        glUseProgram(shaderProgram);
        glDrawArrays(GL_TRIANGLES, 0, boidCount * 3);

        // Swap front and back buffers 
        glfwSwapBuffers(window);

        // Show FPS in the title bar
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
        free(cpuVBO);
        free(boidX);
        free(boidY);
        free(boidDX);
        free(boidDY);
    }
    else
    {
        cudaGraphicsUnregisterResource(resource);
        cudaFree(boidX);
        cudaFree(boidY);
        cudaFree(boidDX);
        cudaFree(boidDY);
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    ParameterManager* parameterManager = static_cast<ParameterManager*>(glfwGetWindowUserPointer(window));
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_V:
            parameterManager->selectVisualRange();
            printParameters(parameterManager);
            break;
        case GLFW_KEY_S:
            parameterManager->selectSeparationFactor();
            printParameters(parameterManager);
            break;
        case GLFW_KEY_C:
            parameterManager->selectCohesionFactor();
            printParameters(parameterManager);
            break;
        case GLFW_KEY_A:
            parameterManager->selectAlignmentFactor();
            printParameters(parameterManager);
            break;
        case GLFW_KEY_UP:
            parameterManager->incrementParameter();
            printParameters(parameterManager);
            break;
        case GLFW_KEY_DOWN:
            parameterManager->decrementParameter();
            printParameters(parameterManager);
            break;
        }
    }
}

// prints the values of parameters to console
void printParameters(const ParameterManager* parameterManager)
{
    Clear();
    std::cout << "Selected Parameter: " << parameterManager->getSelectedName() << std::endl <<
        VISUAL_RANGE_NAME << ": " << parameterManager->getVisualRange() << std::endl <<
        SEPARATION_FACTOR_NAME << ": " << parameterManager->getSeparationFactor() << std::endl <<
        COHESION_FACTOR_NAME << ": " << parameterManager->getCohesionFactor() << std::endl <<
        ALIGNMENT_FACTOR_NAME << ": " << parameterManager->getAlignmentFactor() << std::endl;
}


// Clears the terminal
void Clear()
{
#if defined _WIN32
    system("cls");
#elif defined (__LINUX__) || defined(__gnu_linux__) || defined(__linux__)
    std::cout<< u8"\033[2J\033[1;1H"; // Using ANSI Escape Sequences 
#elif defined (__APPLE__)
    system("clear");
#endif
}



