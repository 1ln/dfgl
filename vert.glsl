#version 330 core

uniform mat4 view;
uniform mat4 projection;
uniform mat4 model;

layout (location = 0) in vec3 pos;
layout (location = 1) in vec2 texCoords;

out vec2 texc;

void main() {

texc = texCoords;
//gl_Position = model * view * projection * vec4(pos,1.);
gl_Position = vec4(pos,1.);
}
