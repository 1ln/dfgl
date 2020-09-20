#version 430 core

uniform mat4 view;
uniform mat4 projection;
uniform mat4 model;

layout (location = 0) out vec2 uv;
layout (location = 0) in vec3 pos;

void main() {

#ifdef ATTTRIBUTELESS

float x = float(((uint(gl_VertexID) + 2u) / 3u) % 2u );
float y = float(((uint(gl_VertexID) + 2u) / 3u) % 2u );

gl_Position = vec4(-1. + x*2.,-1. + y*2.,1.,0.);
uv = vec2(x,y);

#else

//gl_Position = model * view * projection * vec4(pos,1.);
gl_Position = vec4(pos,1.);

#endif

}
