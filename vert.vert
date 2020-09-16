#version 430 core

uniform mat4 view;
uniform mat4 projection;
uniform mat4 model;

in vec4 pos;

int main() {

gl_Position = vec4(model * view * projection * pos,1.);

}
