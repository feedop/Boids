#include "parameterManager.hpp"

void ParameterManager::incrementParameter()
{
	float step = (maxValue - minValue) / STEPS;
	if (*selectedParameter + step > maxValue)
	{
		*selectedParameter = maxValue;
	}
	else
	{
		*selectedParameter += step;
	}
}

void ParameterManager::decrementParameter()
{
	float step = (maxValue - minValue) / STEPS;
	if (*selectedParameter - step < minValue)
	{
		*selectedParameter = minValue;
	}
	else
	{
		*selectedParameter -= step;
	}
}

ParameterManager::ParameterManager(const int boidCount) : boidSize(boidCount <= BOID_SIZE_THRESHHOLD ? LARGE_BOID_SIZE : SMALL_BOID_SIZE), minDistance(boidSize * 1.5f)
{}

void ParameterManager::pause()
{
	paused = !paused;
}

void ParameterManager::selectVisualRange()
{
	selectedParameter = &visualRange;
	minValue = MIN_VISUAL_RANGE;
	maxValue = MAX_VISUAL_RANGE;
	selectedName = VISUAL_RANGE_NAME;
}

void ParameterManager::selectSeparationFactor()
{
	selectedParameter = &separationFactor;
	minValue = MIN_SEPARATION_FACTOR;
	maxValue = MAX_SEPARATION_FACTOR;
	selectedName = SEPARATION_FACTOR_NAME;
}

void ParameterManager::selectCohesionFactor()
{
	selectedParameter = &cohesionFactor;
	minValue = MIN_COHESION_FACTOR;
	maxValue = MAX_COHESION_FACTOR;
	selectedName = COHESION_FACTOR_NAME;
}

void ParameterManager::selectAlignmentFactor()
{
	selectedParameter = &alignmentFactor;
	minValue = MIN_ALIGNMENT_FACTOR;
	maxValue = MAX_ALIGNMENT_FACTOR;
	selectedName = ALIGNMENT_FACTOR_NAME;
}

