#include "boidsGPU.cuh"

#include "cudaGL.h"
#include "cuda_gl_interop.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

#include "defines.h"

namespace GPU
{
	__device__ inline float distance(const float x, const float y)
	{
		return sqrtf(x * x + y * y);
	}

	__device__ void createTrianglesFromPosition(const int id, float* devVBO, float X, float Y, float DX, float DY, const float boidSize)
	{
		float sizeCoefficient = boidSize / distance(DX, DY);

		devVBO[6 * id] = X + sizeCoefficient * DX;
		devVBO[6 * id + 1] = Y + sizeCoefficient * DY;

		devVBO[6 * id + 2] = X - sizeCoefficient * DX - sizeCoefficient * DY;
		devVBO[6 * id + 3] = Y - sizeCoefficient * DY + sizeCoefficient * DX;

		devVBO[6 * id + 4] = X - sizeCoefficient * DX + sizeCoefficient * DY;
		devVBO[6 * id + 5] = Y - sizeCoefficient * DY - sizeCoefficient * DX;
	}

	__global__ void setupCurandStatesKernel(curandState* states, const unsigned long seed, const int boidCount)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= boidCount)
			return;
		curand_init(seed, id, 0, &states[id]);
	}

	__global__ void generateRandomPositionsKernel(curandState* states, float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= boidCount)
			return;

		boidX[id] = curand_uniform(&states[id]) * 2 - 1;
		boidY[id] = curand_uniform(&states[id]) * 2 - 1;
		boidDX[id] = curand_uniform(&states[id]) * 0.02 - 0.01;
		boidDY[id] = curand_uniform(&states[id]) * 0.02 - 0.01;
	}

	// 1. Avoid collisions with other boids
	__device__ void separation(const float X, const float Y, float& DX, float& DY, float* boidX, float* boidY, const int id, const int boidCount,
		float minDistance, float separationFactor)
	{
		float moveX = 0;
		float moveY = 0;

		for (int i = 0; i < boidCount; i++)
		{
			float neighborX = boidX[i];
			float neighborY = boidY[i];

			if (distance(X - neighborX, Y - neighborY) < minDistance)
			{
				moveX += X - neighborX;
				moveY += Y - neighborY;
			}
		}
		DX += moveX * separationFactor;
		DY += moveY * separationFactor;
	}

	// 2. Steer towards the center of the flock (average position of nearby boids)
	__device__ void cohesion(const float X, const float Y, float& DX, float& DY, float* boidX, float* boidY, const int id, const int boidCount,
		float visualRange, float cohesionFactor)
	{
		float moveX = 0;
		float moveY = 0;
		int neighbors = 0;
		for (int i = 0; i < boidCount; i++)
		{
			if (i == id)
				continue;
			float neighborX = boidX[i];
			float neighborY = boidY[i];

			if (distance(X - neighborX, Y - neighborY) < visualRange)
			{
				moveX += neighborX;
				moveY += neighborY;
				neighbors++;
			}
		}
		if (neighbors > 0)
		{
			moveX /= neighbors;
			moveY /= neighbors;
		}

		DX += (moveX - X) * cohesionFactor;
		DY += (moveY - Y) * cohesionFactor;
	}

	// Kernel handling the "separation" and "cohesion" steps of boid behavior
	__global__ void boidSeparationCohesionKernel(float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount,
		float minDistance, float visualRange, float separationFactor, float cohesionFactor)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= boidCount)
			return;

		float X = boidX[id];
		float Y = boidY[id];

		float DX = boidDX[id];
		float DY = boidDY[id];

		// 1. Separation - avoid each other at close range
		separation(X, Y, DX, DY, boidX, boidY, id, boidCount, minDistance, separationFactor);

		// 2. Cohesion - steer towards the center of the flock
		cohesion(X, Y, DX, DY, boidX, boidY, id, boidCount, visualRange, cohesionFactor);

		// Save calculated velocities to global memory
		boidDX[id] = DX;
		boidDY[id] = DY;

	}

	// 3. Match velocity to the average of nearby boids
	__device__ void alignment(const float X, const float Y, float& DX, float& DY, float* boidX, float* boidY, float* boidDX, float* boidDY, const int id,
		const int boidCount, float visualRange, float alignmentFactor)
	{
		float avgDX = 0;
		float avgDY = 0;
		int neighbors = 0;
		for (int i = id + 1; i < boidCount; i++)
		{
			float neighborX = boidX[i];
			float neighborY = boidY[i];

			float dist = distance(X - neighborX, Y - neighborY);
			if (dist < visualRange)
			{
				avgDX += boidDX[i];
				avgDY += boidDY[i];
				neighbors++;
			}
		}
		for (int i = 0; i < id; i++)
		{
			float neighborX = boidX[i];
			float neighborY = boidY[i];

			float dist = distance(X - neighborX, Y - neighborY);
			if (dist < visualRange)
			{
				avgDX += boidDX[i];
				avgDY += boidDY[i];
				neighbors++;
			}
		}
		if (neighbors > 0)
		{
			avgDX /= neighbors;
			avgDY /= neighbors;
		}

		DX += (avgDX - DX) * alignmentFactor;
		DY += (avgDY - DY) * alignmentFactor;
	}

	// Make boids unable to go arbitrarily fast
	__device__ void speedLimit(float& DX, float& DY)
	{
		float speed = distance(DX, DY);
		if (speed > MAX_SPEED)
		{
			DX = DX / speed * MAX_SPEED;
			DY = DY / speed * MAX_SPEED;
		}
		if (speed < MIN_SPEED)
		{
			DX = DX / speed * MIN_SPEED;
			DY = DY / speed * MIN_SPEED;
		}
	}

	// When boids are close to the border, adjust their velocities
	__device__ void keepWithinBounds(const float X, const float Y, float& DX, float& DY)
	{
		if (X < MARGIN - 1)
		{
			DX += TURN_FACTOR;
		}
		if (X > 1 - MARGIN)
		{
			DX -= TURN_FACTOR;
		}
		if (Y < MARGIN - 1)
		{
			DY += TURN_FACTOR;
		}
		if (Y > 1 - MARGIN)
		{
			DY -= TURN_FACTOR;
		}
	}

	// Kernel handling the "alignment" step of boid behavior as well as limiting the boid's speed and keeping it within the bounds of the window
	// Additionally, the function creates triangles for OpenGL based on boid position and the direction of its velocity
	__global__ void boidAlignmentKernel(float* devVBO, float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount,
		float visualRange, float alignmentFactor, float boidSize)
	{
		int id = blockIdx.x * blockDim.x + threadIdx.x;
		if (id >= boidCount)
			return;

		float X = boidX[id];
		float Y = boidY[id];
		float DX = boidDX[id];
		float DY = boidDY[id];

		// 3. Alignment - match the average velocity of nearby boids
		alignment(X, Y, DX, DY, boidX, boidY, boidDX, boidDY, id, boidCount, visualRange, alignmentFactor);

		// 4. Speed limit
		speedLimit(DX, DY);

		// 5. Keep within bounds
		keepWithinBounds(X, Y, DX, DY);

		// Save new positions
		X += DX;
		Y += DY;

		boidDX[id] = DX;
		boidDY[id] = DY;

		boidX[id] = X;
		boidY[id] = Y;

		// Calculate triangle vertices for OpenGL
		createTrianglesFromPosition(id, devVBO, X, Y, DX, DY, boidSize);
	}

	// Allocate buffers for boid positions and velocities
	void initializeBoidLists(float** boidX, float** boidY, float** boidDX, float** boidDY, const int boidCount)
	{
		HANDLE_ERROR(cudaMalloc((void**)boidX, boidCount * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)boidY, boidCount * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)boidDX, boidCount * sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)boidDY, boidCount * sizeof(float)));
	}

	// Generate initial positions and velocities of boids
	void generateRandomPositions(float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount)
	{
		int threadsPerBlock = boidCount > 1024 ? 1024 : boidCount;
		int blocks = (boidCount + threadsPerBlock - 1) / threadsPerBlock;

		// Set up random seeds
		curandState* devStates;
		cudaMalloc(&devStates, boidCount * sizeof(curandState));
		srand(static_cast<unsigned int>(time(0)));
		int seed = rand();
		setupCurandStatesKernel << <blocks, threadsPerBlock >> > (devStates, seed, boidCount);

		// Generate random positions and velocity vectors
		float* devVBO = 0;
		size_t numBytes;
		
		generateRandomPositionsKernel << <blocks, threadsPerBlock >> > (devStates, boidX, boidY, boidDX, boidDY, boidCount);

		cudaFree(devStates);
	}

	// Calculate boid positions for the next frame
	void calculatePositions(cudaGraphicsResource_t* resource, float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount, const ParameterManager& parameterManager)
	{
		int threadsPerBlock = boidCount > 1024 ? 1024 : boidCount;
		int blocks = (boidCount + threadsPerBlock - 1) / threadsPerBlock;		

		// 1. Separation - avoid each other,
		// 2. Cohesion - steer towards the center of the flock
		boidSeparationCohesionKernel << <blocks, threadsPerBlock>> > (boidX, boidY, boidDX, boidDY, boidCount,
			parameterManager.minDistance, parameterManager.getVisualRange(), parameterManager.getSeparationFactor(), parameterManager.getCohesionFactor());

		// Map VBO to CUDA
		float* devVBO = 0;
		size_t numBytes;

		HANDLE_ERROR(cudaGraphicsMapResources(1, resource, 0));
		HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devVBO, &numBytes, *resource));

		// 3. Alignment - match the average velocity of nearby boids,
		// 4. Speed limit,
		// 5. Keep within bounds
		boidAlignmentKernel << <blocks, threadsPerBlock>> > (devVBO, boidX, boidY, boidDX, boidDY, boidCount,
			parameterManager.getVisualRange(), parameterManager.getAlignmentFactor(), parameterManager.boidSize);
		
		HANDLE_ERROR(cudaGraphicsUnmapResources(1, resource, 0));
	}
}