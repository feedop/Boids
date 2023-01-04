#pragma once

#define DEFAULT_BOIDCOUNT 500

#define WIDTH 1200
#define HEIGHT 800

#define VERT_FILE "Shaders/default.vert"
#define FRAG_FILE "Shaders/default.frag"

#define MARGIN 0.03f
#define TURN_FACTOR 0.002f

#define STEPS 20

#define BOID_SIZE_THRESHHOLD 500
#define SMALL_BOID_SIZE 0.01f
#define LARGE_BOID_SIZE 0.02f

#define MIN_VISUAL_RANGE 0.05f
#define INITIAL_VISUAL_RANGE 0.2f
#define MAX_VISUAL_RANGE 0.2f

#define MIN_SEPARATION_FACTOR 0.005f
#define INITIAL_SEPARATION_FACTOR 0.02f
#define MAX_SEPARATION_FACTOR 0.1f

#define MIN_COHESION_FACTOR 0.0001f
#define INITIAL_COHESION_FACTOR 0.001f
#define MAX_COHESION_FACTOR 0.005f

#define MIN_ALIGNMENT_FACTOR 0.001f
#define INITIAL_ALIGNMENT_FACTOR 0.008f
#define MAX_ALIGNMENT_FACTOR 0.02f

#define MIN_SPEED 0.004f
#define MAX_SPEED 0.015f