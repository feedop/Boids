#pragma once

#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>

void initializeBuffer(GLuint* VAO, GLuint* VBO, const int boidCount);

void initializeShaders(GLuint& vertexShader, GLuint& fragmentShader, GLuint& shaderProgram);

std::string getFileContent(const char* filename);