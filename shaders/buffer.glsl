#version 330     

out vec4 FragColor; 

uniform vec2 resolution;
uniform int seed;

#define SINE

#ifdef SINE

float h21(vec2 p) {
    return fract(sin(dot(p,vec2(12.9898,78.233)))*float(seed));
}

#else

float h21(vec2 p) {
    uvec2 n = uvec2(ivec2(p))*uvec2(1391674541U,seed);
    uint h = (n.x^n.y) * 1391674541U;
    return float(h) * (1./float(0xffffffffU));
}

#endif 
  
void main() { 

vec2 uv = (2.* (gl_FragCoord.xy) - resolution.xy)/resolution.y;

uv *= 10.;

vec2 loc = vec2(floor(uv));
FragColor = vec4(vec3(h21(loc)),1.0);

}
