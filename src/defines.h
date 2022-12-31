#pragma once

// MAX = 7168
#define DEFAULT_BOIDCOUNT 10000

#define WIDTH 1200
#define HEIGHT 800

#define BOID_SIZE 0.005f

#define VERT_FILE "Shaders/default.vert"
#define FRAG_FILE "Shaders/default.frag"

#define VISUAL_RANGE 0.1f

#define MARGIN 0.05f
#define TURN_FACTOR 0.003f

#define MIN_DISTANCE 0.012f
#define SEPARATION_FACTOR 0.015f

#define COHESION_FACTOR 0.002f;

#define ALIGNMENT_FACTOR 0.005f;

#define MIN_SPEED 0.002f
#define MAX_SPEED 0.015f