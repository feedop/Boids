#pragma once
#include "defines.h"

class ParameterManager
{
private:
	float visualRange = INITIAL_VISUAL_RANGE;
	float separationFactor = INITIAL_SEPARATION_FACTOR;
	float cohesionFactor = INITIAL_COHESION_FACTOR;
	float alignmentFactor = INITIAL_ALIGNMENT_FACTOR;

	void incrementParameter(float& field, const float minValue, const float maxValue);
	void decrementParameter(float& field, const float minValue, const float maxValue);

public:
	const float boidSize;
	const float minDistance;

	ParameterManager(const int boidCount);

	void incrementVisualRange();
	void incrementSeparationFactor();
	void incrementCohesionFactor();
	void incrementAlignmentFactor();

	void decrementVisualRange();
	void decrementSeparationFactor();
	void decrementCohesionFactor();
	void decrementAlignmentFactor();

	float getVisualRange() const
	{
		return visualRange;
	}

	float getSeparationFactor() const
	{
		return separationFactor;
	}

	float getCohesionFactor() const
	{
		return cohesionFactor;
	}

	float getAlignmentFactor() const
	{
		return alignmentFactor;
	}
};