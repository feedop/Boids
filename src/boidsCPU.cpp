#include "boidsCPU.hpp"

#include <algorithm>
#include <random>
#include <vector>

#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace CPU
{
	inline float distance(const float x, const float y)
	{
		return sqrtf(x * x + y * y);
	}

	void createTrianglesFromPosition(float* cpuVBO, float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount, const float boidSize)
	{
		for (int i = 0; i < boidCount; i++)
		{
			float sizeCoefficient = boidSize / distance(boidDX[i], boidDY[i]);

			cpuVBO[6 * i] = boidX[i] + sizeCoefficient * boidDX[i];
			cpuVBO[6 * i + 1] = boidY[i] + sizeCoefficient * boidDY[i];

			cpuVBO[6 * i + 2] = boidX[i] - sizeCoefficient * boidDX[i] - sizeCoefficient * boidDY[i];
			cpuVBO[6 * i + 3] = boidY[i] - sizeCoefficient * boidDY[i] + sizeCoefficient * boidDX[i];

			cpuVBO[6 * i + 4] = boidX[i] - sizeCoefficient * boidDX[i] + sizeCoefficient * boidDY[i];
			cpuVBO[6 * i + 5] = boidY[i] - sizeCoefficient * boidDY[i] - sizeCoefficient * boidDX[i];
		}
		// copy triangle data to OpenGL buffer
		glBufferSubData(GL_ARRAY_BUFFER, 0, 6 * boidCount * sizeof(float), cpuVBO);
	}

	// Ugly code but it makes reusing the cuda code on CPU easier
	void initializeBoidLists(float** cpuVBO, float** boidX, float** boidY, float** boidDX, float** boidDY, const int boidCount)
	{
		*cpuVBO = new float[6 * boidCount];

		*boidX = new float[boidCount];
		*boidY = new float[boidCount];
		*boidDX = new float[boidCount];
		*boidDY = new float[boidCount];
	}

	void generateRandomPositions(float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount)
	{
		std::random_device rd;
		std::default_random_engine randomEngine(rd());
		std::uniform_real_distribution<float> positionDistribution(-1, 1);
		std::uniform_real_distribution<float> velocityDistribution(-0.01f, 0.01f);

		std::generate(boidX, boidX + boidCount, [&]() { return positionDistribution(randomEngine); });
		std::generate(boidY, boidY + boidCount, [&]() { return positionDistribution(randomEngine); });

		std::generate(boidDX, boidDX + boidCount, [&]() { return velocityDistribution(randomEngine); });
		std::generate(boidDY, boidDY + boidCount, [&]() { return velocityDistribution(randomEngine); });

	}

	// 1. Avoid collisions with other boids
	void separation(float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount, float minDistance, float visualRange, float separationFactor)
	{
		for (int i = 0; i < boidCount; i++)
		{
			float X = boidX[i];
			float Y = boidY[i];
			float moveX = 0;
			float moveY = 0;
			for (int j = 0; j < boidCount; j++)
			{
				float neighborX = boidX[j];
				float neighborY = boidY[j];

				if (distance(X - neighborX, Y - neighborY) < minDistance)
				{
					moveX += X - neighborX;
					moveY += Y - neighborY;
				}
			}
			boidDX[i] += moveX * separationFactor;
			boidDY[i] += moveY * separationFactor;
		}
	}

	// 2. Steer towards the center of the flock (average position of nearby boids)
	void cohesion(float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount, float visualRange, float cohesionFactor)
	{
		for (int i = 0; i < boidCount; i++)
		{
			float X = boidX[i];
			float Y = boidY[i];
			float moveX = 0;
			float moveY = 0;
			int neighbors = 0;
			for (int j = 0; j < boidCount; j++)
			{
				float neighborX = boidX[j];
				float neighborY = boidY[j];

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

			boidDX[i] += (moveX - X) * cohesionFactor;
			boidDY[i] += (moveY - Y) * cohesionFactor;
		}
	}

	// 3. Match velocity to the average of nearby boids
	void alignment(float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount, float visualRange, float alignmentFactor)
	{
		std::vector<float> tempDX(boidCount);
		std::vector<float> tempDY(boidCount);

		for (int i = 0; i < boidCount; i++)
		{
			float X = boidX[i];
			float Y = boidY[i];
			float DX = boidDX[i];
			float DY = boidDY[i];

			float avgDX = 0;
			float avgDY = 0;
			int neighbors = 0;
			for (int j = 0; j < boidCount; j++)
			{
				float neighborX = boidX[j];
				float neighborY = boidY[j];

				if (distance(X - neighborX, Y - neighborY) < visualRange)
				{
					avgDX += boidDX[j];
					avgDY += boidDY[j];
					neighbors++;
				}
			}
			if (neighbors > 0)
			{
				avgDX /= neighbors;
				avgDY /= neighbors;
			}

			tempDX[i] += (avgDX - DX) * alignmentFactor;
			tempDY[i] += (avgDY - DY) * alignmentFactor;
		}

		// Update velocities from temp array
		for (int i = 0; i < boidCount; i++)
		{
			boidDX[i] += tempDX[i];
			boidDY[i] += tempDY[i];
		}
	}

	// Make boids unable to go arbitrarily fast
	void speedLimit(float* boidDX, float* boidDY, const int boidCount)
	{
		for (int i = 0; i < boidCount; i++)
		{
			float speed = distance(boidDX[i], boidDY[i]);
			if (speed > MAX_SPEED)
			{
				boidDX[i] = boidDX[i] / speed * MAX_SPEED;
				boidDY[i] = boidDY[i] / speed * MAX_SPEED;
			}
			if (speed < MIN_SPEED)
			{
				boidDX[i] = boidDX[i] / speed * MIN_SPEED;
				boidDY[i] = boidDY[i] / speed * MIN_SPEED;
			}
		}
	}

	// When boids are close to the border, adjust their velocities
	void keepWithinBounds(float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount)
	{
		for (int i = 0; i < boidCount; i++)
		{
			if (boidX[i] < MARGIN - 1)
			{
				boidDX[i] += TURN_FACTOR;
			}
			if (boidX[i] > 1 - MARGIN)
			{
				boidDX[i] -= TURN_FACTOR;
			}
			if (boidY[i] < MARGIN - 1)
			{
				boidDY[i] += TURN_FACTOR;
			}
			if (boidY[i] > 1 - MARGIN)
			{
				boidDY[i] -= TURN_FACTOR;
			}
		}
	}

	// In normal code these parameters would be std::vector, but I've decided to keep the raw pointers in order to reuse the cuda code
	void calculatePositions(float* cpuVBO, float* boidX, float* boidY, float* boidDX, float* boidDY, const int boidCount, const ParameterManager& parameterManager)
	{
		float visualRange = parameterManager.getVisualRange();

		// 1. Separation - avoid each other at close range
		separation(boidX, boidY, boidDX, boidDY, boidCount, parameterManager.minDistance, visualRange, parameterManager.getSeparationFactor());

		// 2. Cohesion - steer towards the center of the flock
		cohesion(boidX, boidY, boidDX, boidDY, boidCount, visualRange, parameterManager.getCohesionFactor());

		// 3. Alignment - match the average velocity of nearby boids
		alignment(boidX, boidY, boidDX, boidDY, boidCount, visualRange, parameterManager.getAlignmentFactor());

		// 4. Speed limit
		speedLimit(boidDX, boidDY, boidCount);

		// 5. Keep within bounds
		keepWithinBounds(boidX, boidY, boidDX, boidDY, boidCount);

		// Update
		for (int i = 0; i < boidCount; i++)
		{
			boidX[i] += boidDX[i];
			boidY[i] += boidDY[i];
		}

		// Create traingles for OpenGL
		createTrianglesFromPosition(cpuVBO, boidX, boidY, boidDX, boidDY, boidCount, parameterManager.boidSize);

	}
}
