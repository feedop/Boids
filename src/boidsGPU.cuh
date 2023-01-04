#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "parameterManager.hpp"

#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace GPU
{
	void initializeBoidLists(float** boidX, float** boidY, float** boidDX, float** boidDY, const int boidCount);

	void generateRandomPositions(GLuint VBO, float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount);

	void calculatePositions(GLuint VBO, float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount, const ParameterManager& parameterManager);
}