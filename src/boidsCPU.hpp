#pragma once

#include "parameterManager.hpp"

namespace CPU
{
	void initializeBoidLists(float** cpuVBO, float** boidX, float** boidY, float** boidDX, float** boidDY, const int boidCount);

	void generateRandomPositions(float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount);

	void calculatePositions(float* cpuVBO, float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount, const ParameterManager& parameterManager);
}