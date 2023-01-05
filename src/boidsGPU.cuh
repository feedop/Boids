#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "parameterManager.hpp"

#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

static void HandleError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ));

namespace GPU
{
	void initializeBoidLists(float** boidX, float** boidY, float** boidDX, float** boidDY, const int boidCount);

	void generateRandomPositions(cudaGraphicsResource_t* resource, float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount);

	void calculatePositions(cudaGraphicsResource_t* resource, float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount, const ParameterManager& parameterManager);
}