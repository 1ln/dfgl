#version 330     

// dolson
out vec4 FragColor; 

uniform vec2 resolution;

uniform float time;

uniform vec3 camPos;
uniform sampler2D tex;
uniform int seed;

uniform vec2 mouse;

float h11(float p) {
    return fract(sin(p)*float(43758.5453+seed));
}

float h21(vec2 p) {
    return fract(sin(dot(p,vec2(12.9898,78.233)*float(43758.5453+seed))));
}

vec2 h22(vec2 p) {
    return fract(vec2(sin(p.x*353.64+p.y*135.1),cos(p.x*333.76+p.y*57.33)));
}

vec2 h12rad(float n) {
    float a = fract(sin(n*5673.)*48.)*radians(180.);
    float b = fract(sin(a+n)*6446.)*float(seed);
    return vec2(cos(a),sin(b));
}

float spiral(vec2 p,float n,float h) {
     float ph = pow(length(p),1./n)*32.;
     p *= mat2(cos(ph),sin(ph),sin(ph),-cos(ph));
     return h-length(p) / 
     sin((atan(p.x,-p.y)+radians(180.)/(radians(180.)*2.))*radians(180.)); 
}

float concentric(vec2 p,float h) {
    return cos(length(p))-h;
}

float julia(vec2 p,float b,float f) {
    float k = 0.;
    for(int i = 0; i < 64; i++) {
        p = vec2((p.x*p.x-p.y*p.y)*.5,(p.x*p.y))-f;
        k += .05;
    
        if(dot(p,p) > b) {
            break;
        }
    return k;
    }
}

vec2 diag(vec2 uv) {
   vec2 r = vec2(0.);
   r.x = 1.1547 * uv.x;
   r.y = uv.y + .5 * r.x;
   return r;
}

float sin2(vec2 p,float h) {
    return sin(p.x*h) * sin(p.y*h);
}

float expStep(float x,float k) {
    return exp((x*k)-k);
}

vec3 fmCol(float t,vec3 a,vec3 b,vec3 c,vec3 d) {
    return a + b * cos((radians(180.)*2.0) * (c * t + d));
}

vec3 rgbHsv(vec3 c) {
    vec3 rgb = clamp(abs(
    mod(c.x * 6. + vec3(0.,4.,2.),6.)-3.)-1.,0.,1.);

    rgb = rgb * rgb * (3. - 2. * rgb);
    return c.z * mix(vec3(1.),rgb,c.y);
}

vec3 wbar(vec2 uv,vec3 fcol,vec3 col,float y,float h) {
    return mix(fcol,col,step(abs(uv.y-y),h));
}

float smou(float d1,float d2,float k) {
    float h = clamp(0.5 + 0.5 * (d2-d1)/k,0.0,1.0);
    return mix(d2,d1,h) - k * h * (1.0 - h);
}

float smod(float d1,float d2,float k) {
    float h = clamp(0.5 - 0.5 * (d2+d1)/k,0.0,1.0);
    return mix(d2,-d1,h) + k * h * (1.0 - h);
}

float smoi(float d1,float d2,float k) {

    float h = clamp(0.5 + 0.5 * (d2-d1)/k,0.0,1.0);
    return mix(d2,d1,h) + k * h * (1.0 - h);

}

float cell(vec2 x,float n) {
    x *= n;
    vec2 p = floor(x);
    vec2 f = fract(x);
    float min_dist = 1.;
    
    for(int i = -1; i <= 1; i++) {
        for(int j = -1; j <= 1; j++) {

        vec2 b = vec2(float(j),float(i));
        vec2 r = h22(p+b);
        vec2 diff = (b+r-f);
        float d = length(diff);
        min_dist = min(min_dist,d);
        }
    }
    return min_dist;
}

float f6(vec2 p) {
    float f = 1.;
    mat2 m = mat2(.8,.6,-.6,.8);

    f += .5    * sin2(p,1.); p = m*p*2.01;
    f += .25   * sin2(p,1.); p = m*p*2.04;
    f += .125  * sin2(p,1.); p = m*p*2.1;
    f += .0625 * sin2(p,1.); p = m*p*2.12;
    f += .03125 * sin2(p,1.); p = m*p*2.25;
    f += .015625 * sin2(p,1.);

    return f / .94;
}

float dd(vec2 p) {
    vec2 q = vec2(f6(p+vec2(0.,1.)),
                  f6(p+vec2(4.,2.)));

    vec2 r = vec2(f6(p + 4. * q + vec2(4.5,2.3)),
                  f6(p + 4. * q + vec2(2.25,5.)));

    return f6(p + 4. * r);
}

mat2 rot(float a) {

    float c = cos(a);
    float s = sin(a);
    
    return mat2(c,-s,s,c);
}

float circle(vec2 p,float r) {
    return length(p) - r;
}

float ring(vec2 p,float r,float w) {
    return abs(length(p) - r) - w;
}

float eqTriangle(vec2 p,float r) { 

     const float k = sqrt(3.);
   
     p.x = abs(p.x) - 1.;
     p.y = p.y + 1./k;

     if(p.x + k * p.y > 0.) {
         p = vec2(p.x - k * p.y,-k * p.x - p.y)/2.;
     }

     p.x -= clamp(p.x,-2.,0.);
     return -length(p) * sign(p.y);    

}
 
float rect(vec2 p,vec2 b) {
    vec2 d = abs(p)-b;
    return length(max(d,0.)) + min(max(d.x,d.y),0.);
}

void main() { 
vec3 c = vec3(0.);

vec2 uv = (gl_FragCoord.xy -.5* resolution.xy)/resolution.y;
vec2 p = uv;
//p *= 5.;
p *= .5;

float n = dd(p); 

float d = 1.;
float sq = rect(p,vec2(.25));
float cir = circle(p,1.);

float jl = julia(p,100.,n);

d = min(d,jl);
c = vec3(d,.25,.5);

c = pow(c,vec3(.4545));
FragColor = vec4(c,1.0);
 

}
