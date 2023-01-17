# Boids

Author: Filip Skrzeczkowski

A program for visualization of fish shoal behavior by implementing the boid algorithm using CUDA for computing and OpenGL for graphics.

## How to build?

The package contains a Visual Studio 2022 .sln file. Use it to build the program.

## How to run?

Use ./Boids [-c] [boid count].

Both parameters are optional and the order is not important.

"-c" switches from the default CUDA implementation to standard CPU computing, which results in significantly worse performance.

"boid count" is responsible for how many boids are going to be simulated. The default value is 10000.
If you set this parameter to 800 or less, the rendered boids are going to be bigger.

## How to use?

After launching, two windows should appear:
1. The graphical GLFW windows containing the simulation and the FPS count (in the tile bar)
2. A console window that should display simulation parameters (initialized by default value).

To change the parameters, **the OpenGL window should be focused**. Firstly, use V, S, C and A keys
(corresponding to "Visual range", "Separation factor", "Cohesion Factor" and "Alignment Factor respectively)
to select the parameter to change. Next, use the arrow keys to increase/decrease the selected parameter.
There are always lower and upper bounds for these factors.
