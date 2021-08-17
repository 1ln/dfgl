#version 330 core     

//dolson
//2020

out vec4 FragColor;

uniform vec2 resolution;
uniform float time;
uniform int seed; 

mat2 m = mat2(0.8,0.6,-0.6,0.8);

 

float h(float p) {

return fract(sin(p)*43578.5453);

}

float n(vec2 p) {
return sin(p.x)*sin(p.y);
}

float f6(vec2 p) {

    float f = 1.;

    f += 0.5      * n(p); p = m*p*2.01;
    f += 0.25     * n(p); p = m*p*2.02;
    f += 0.125    * n(p); p = m*p*2.04;
    f += 0.0625   * n(p); p = m*p*2.03;
    f += 0.0325   * n(p); p = m*p*2.06;
    f += 0.015625 * n(p);
    return f/0.92;    

}

void main() {
 
    vec3 c;
    vec2 uv = gl_FragCoord.xy / resolution.xy; 
         uv = (gl_FragCoord.xy-.5*resolution)/resolution.y;  

    float scl = 10.;
    uv *= scl;
    
    vec2 loc = floor(uv);
    vec2 fr  = fract(uv);
    
    c = vec3(h(loc.x+h(loc.y)));
    
    float x,y;
    x = floor(h(loc.x)*(scl-1.)); y = floor(h(loc.y)*(scl-1.));

    c = mix(c,vec3(1.,0.,0.),
    step(abs(loc.x-x),0.)*step(abs(loc.y-y),0.));

    FragColor = vec4(c,1.);

}
