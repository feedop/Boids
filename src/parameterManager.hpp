#pragma once
#include <string>

#include "defines.h"

// This class stores the values of basic boid behavior parameters and manages increasing/descreasing them
class ParameterManager
{
private:
	float visualRange = INITIAL_VISUAL_RANGE;
	float separationFactor = INITIAL_SEPARATION_FACTOR;
	float cohesionFactor = INITIAL_COHESION_FACTOR;
	float alignmentFactor = INITIAL_ALIGNMENT_FACTOR;

	float* selectedParameter = &visualRange;
	float minValue = MIN_VISUAL_RANGE;
	float maxValue = MAX_VISUAL_RANGE;
	std::string selectedName = "Visual Range";

public:
	const float boidSize;
	const float minDistance;

	ParameterManager(const int boidCount);

	void incrementParameter();
	void decrementParameter();

	void selectVisualRange();
	void selectSeparationFactor();
	void selectCohesionFactor();
	void selectAlignmentFactor();

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

	std::string getSelectedName() const
	{
		return selectedName;
	}
};