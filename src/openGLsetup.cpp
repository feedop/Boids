#include "openGLsetup.hpp"

#include "defines.h"

// Read a file into a string (used for compiling shaders)
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
    else // fallback 
    {
        char fallbackFilename[128];
        sprintf_s(fallbackFilename, "../../%s", filename);
        std::ifstream in(fallbackFilename, std::ios::binary);
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
    }
    throw std::system_error(errno, std::generic_category(), filename);
}

// Read a shader, compile it and return its id
GLuint createShader(GLuint type, const std::string source)
{
    const char* source_cstr = source.c_str();
    int result;
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source_cstr, NULL);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &result);

    // Error handling
    if (result == GL_FALSE)
    {
        int length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        char* message = (char*)_malloca(length * sizeof(char));
        glGetShaderInfoLog(shader, length, &length, message);
        std::cout << "Failed to compile " << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader" << std::endl;
        std::cout << message << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

// Set up shaders and attach them to a new shader program, wchich is then linked
void initializeShaders(GLuint& vertexShader, GLuint& fragmentShader, GLuint& shaderProgram)
{
    // Read vertex and fragment shaders
    std::string vertexSource;
    std::string fragmentSource;
    try
    {
        vertexSource = getFileContent(VERT_FILE);
        fragmentSource = getFileContent(FRAG_FILE);
    }
    catch (std::system_error& error)
    {
        std::cout << "Error reading shader files. " << error.what() << std::endl;
        exit(1);
    }

    // Create shaders
    vertexShader = createShader(GL_VERTEX_SHADER, vertexSource);
    fragmentShader = createShader(GL_FRAGMENT_SHADER, fragmentSource);

    // Create a shader program, attach the shaders to it nad then link it to gl
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);

    glLinkProgram(shaderProgram);

    // Error handling
    int result;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &result);
    if (result != GL_TRUE)
    {
        GLint length;
        glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &length);
        char* message = (char*)_malloca(length * sizeof(char));
        glGetProgramInfoLog(shaderProgram, length, &length, message);
        std::cerr << "Failed to link the shader program!" << std::endl;
        std::cerr << message << std::endl;
    } 

    // Delete unnecessary objects
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

// create a VAO - the scene and VBO - the boid position buffer
void initializeBuffer(GLuint* VAO, GLuint* VBO, const int boidCount)
{
    // Generate buffers
    glGenVertexArrays(1, VAO);
    glGenBuffers(1, VBO);

    // Bind
    glBindVertexArray(*VAO);
    glBindBuffer(GL_ARRAY_BUFFER, *VBO);

    // Set initial data to NULL - it will be filled later
    glBufferData(GL_ARRAY_BUFFER, 6 * boidCount * sizeof(float), NULL, GL_STREAM_DRAW);

    // Set vertex attribute layout
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}