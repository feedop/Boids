#include "parameterManager.hpp"

void ParameterManager::incrementParameter(float& parameter, const float minValue, const float maxValue)
{
	float step = (maxValue - minValue) / STEPS;
	if (parameter + step > maxValue)
	{
		parameter = maxValue;
	}
	else
	{
		parameter += step;
	}
}

void ParameterManager::decrementParameter(float& parameter, const float minValue, const float maxValue)
{
	float step = (maxValue - minValue) / STEPS;
	if (parameter - step < minValue)
	{
		parameter = minValue;
	}
	else
	{
		parameter -= step;
	}
}

ParameterManager::ParameterManager(const int boidCount) : boidSize(boidCount <= BOID_SIZE_THRESHHOLD ? LARGE_BOID_SIZE : SMALL_BOID_SIZE), minDistance(boidSize * 1.5)
{}

void ParameterManager::incrementVisualRange()
{
	incrementParameter(visualRange, MIN_VISUAL_RANGE, MAX_VISUAL_RANGE);
}

void ParameterManager::incrementSeparationFactor()
{
	incrementParameter(separationFactor, MIN_SEPARATION_FACTOR, MAX_SEPARATION_FACTOR);
}

void ParameterManager::incrementCohesionFactor()
{
	incrementParameter(cohesionFactor, MIN_COHESION_FACTOR, MAX_COHESION_FACTOR);
}

void ParameterManager::incrementAlignmentFactor()
{
	incrementParameter(alignmentFactor, MIN_ALIGNMENT_FACTOR, MAX_ALIGNMENT_FACTOR);
}

void ParameterManager::decrementVisualRange()
{
	decrementParameter(visualRange, MIN_VISUAL_RANGE, MAX_VISUAL_RANGE);
}

void ParameterManager::decrementSeparationFactor()
{
	decrementParameter(separationFactor, MIN_SEPARATION_FACTOR, MAX_SEPARATION_FACTOR);
}

void ParameterManager::decrementCohesionFactor()
{
	decrementParameter(cohesionFactor, MIN_COHESION_FACTOR, MAX_COHESION_FACTOR);
}

void ParameterManager::decrementAlignmentFactor()
{
	decrementParameter(alignmentFactor, MIN_ALIGNMENT_FACTOR, MAX_ALIGNMENT_FACTOR);
}
