#include "openGLsetup.hpp"

#include <fstream>
#include <sstream>
#include <cerrno>
#include <iostream>

#include "defines.h"

std::string getFileContent(const char* filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (in)
    {
        std::string contents;
        in.seekg(0, std::ios::end);
        contents.resize(in.tellg());
        in.seekg(0, std::ios::beg);
        in.read(&contents[0], contents.size());
        in.close();
        return contents;
    }
    throw errno;
}

GLuint createShader(GLuint type, const std::string source)
{
    const char* source_cstr = source.c_str();
    int result;
    GLuint shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(shader, 1, &source_cstr, NULL);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE)
    {
        int length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        char* message = (char*)alloca(length * sizeof(char));
        glGetShaderInfoLog(shader, length, &length, message);
        std::cout << "Failed to compile " << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader" << std::endl;
        std::cout << message << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

void initializeShaders(GLuint& vertexShader, GLuint& fragmentShader, GLuint& shaderProgram)
{
    std::string vertexSource = getFileContent(VERT_FILE);
    std::string fragmentSource = getFileContent(FRAG_FILE);

    vertexShader = createShader(GL_VERTEX_SHADER, vertexSource);
    fragmentShader = createShader(GL_FRAGMENT_SHADER, fragmentSource);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);

    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

void initializeBuffer(GLuint* VAO, GLuint* VBO, const int boidCount)
{
    glGenVertexArrays(1, VAO);
    glGenBuffers(1, VBO);

    glBindVertexArray(*VAO);
    glBindBuffer(GL_ARRAY_BUFFER, *VBO);
    glBufferData(GL_ARRAY_BUFFER, boidCount * 6 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    // Set vertex attribute layout
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}