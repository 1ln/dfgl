#include "Shader.h"

Shader::Shader(const char* vertPath,const char* fragPath) {

    std::string vertCode;
    std::string fragCode;

    std::ifstream vShaderFile;
    std::ifstream fShaderFile;

    vShaderFile.exception (std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exception (std::ifstream::failbit | std::ifstream::badbit);

    try {

        vShaderFile.open(vertPath);
        fShaderFile.open(fragPath);

        std::stringstream vShaderStream, fShaderStream;

        vShaderStream << vShaderFile.rdbuf();
        fShaderStream << fShaderFile.rdbuf();

        vShaderFile.close();
        fShaderFile.close();

        vertCode = vShaderStream.str();
        fragCode = fShaderStream.str();

    }

    catch(std::ifstream::failure& e) {
       std::cout << "Error, shader could not be read" << std::endl;
    }

    const char* vShaderCode = vertCode.c_str();
    const char* fShaderCode = fragCode.c_str();

    unsigned int vert,frag;

    vert = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vert,1,&vShaderCode,NULL);
    glCompileShader(vert);
    checkCompileErrors(vert,"VERTEX");

    frag = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(frag,1,&fShaderCode,NULL);
    glCompileShader(frag);
    checkCompileErrors(frag,"FRAGMENT");

    id = glCreateProgram();
    glAttachShader(id,vert);
    glAttachShader(id,frag);

    glLinkProgram(id);
    checkCompileErrors(id,"PROGRAM");
    
    glDeleteShader(vert);
    glDeletShader(frag);

    }








   


