#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class Shader {

public:

Shader(const char* vertPath,const char* fragPath); 

void use();

void setBool(const std::string &name,bool value) const;
void setInt(const std::string &name,int value) const;
void setTex(const std::string &name,int value) const;
void setFloat(const std::string &name,float value) const;

void setVec2(const std::string &name,
             int count,
             glm::vec2 const& v) const;

void setMat4(const std::string &name,
             int count,
             glm::mat4 const& m4) const;

unsigned int id;

private:

void checkCompileErrors(unsigned int shader,std::string type);

};
