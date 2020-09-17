#version 430 core

uniform mat4 view;
uniform mat4 projection;
uniform mat4 model;

layout (location = 0) in vec3 pos;

void main() {

gl_Position = vec4(model * view * projection * pos,1.);

}
