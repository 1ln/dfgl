#version 430 core

// dolson,2019

out vec4 FragColor;

uniform vec2 resolution;
uniform float time;

const float PI   =  radians(180.0); 
const float PI2  =  PI * 2.;

float seed = 1525267.;

float hash(float p) {
    return fract(sin(p) * 4358.5453);
}

float hash(vec2 p) {
   return fract(sin(dot(p.xy,vec2(12.9898,78.233)))*4358.5353);
}

vec2 mod289(vec2 p) { return p - floor(p * (1. / 289.)) * 289.; }
vec3 mod289(vec3 p) { return p - floor(p * (1. / 289.)) * 289.; }
vec3 permute(vec3 p) { return mod289(((p * 34.) + 1.) * p); } 

float ns2(vec2 p) {

    const float k1 = (3. - sqrt(3.))/6.;
    const float k2 = .5 * (sqrt(3.) -1.);
    const float k3 = -.5773;
    const float k4 = 1./41.;

    const vec4 c = vec4(k1,k2,k3,k4);
    
    vec2 i = floor(p + dot(p,c.yy));
    vec2 x0 = p - i + dot(i,c.xx);
  
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.,0.) : vec2(0.,1.);
    vec4 x12 = x0.xyxy + c.xxzz;
    x12.xy -= i1;

    i = mod289(i);
    
    vec3 p1 = permute(permute(i.y + vec3(0.,i1.y,1.))
        + i.x + vec3(0.,i1.x,1.));
  
    p1 = permute(mod289(p1 + vec3(float(seed))));

    vec3 m = max(.5 - 
    vec3(dot(x0,x0),dot(x12.xy,x12.xy),dot(x12.zw,x12.zw)),0.);
    m = m * m; 
    m = m * m;

    vec3 x = fract(p1 * c.www) - 1.;
    vec3 h = abs(x) - .5;
    vec3 ox = floor(x + .5);
    vec3 a0 = x - ox; 
    m *= 1.792842 - 0.853734 * (a0 * a0 + h * h);
     
    vec3 g;
    g.x = a0.x * x0.x + h.x * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130. * dot(m,g);
}

float f(vec2 x,int octaves) {

    float f = 0.;

    for(int i = 1; i < octaves; i++) {
 
    float e = pow(2.,float(i));
    float s = (1./e);
    f += ns2(x*e)*s;   
    
    }    

    return f * .5 + .5;
}

float sin2(vec2 p,float h) {
    
    return sin(p.x*h) * sin(p.y*h);
}

float fib(float n) {

    return pow(( 1. + sqrt(5.)) /2.,n) -
           pow(( 1. - sqrt(5.)) /2.,n) / sqrt(5.); 

}

float envImp(float x,float k) {

    float h = k * x;
    return h * exp(1.0 - h);
}

float envSt(float x,float k,float n) {

    return exp(-k * pow(x,n));

}

float cubicImp(float x,float c,float w) {

    x = abs(x - c);
    if( x > w) { return 0.0; }
    x /= w;
    return 1.0 - x * x  * (3.0 - 2.0 * x);

}

float sincPh(float x,float k) {

    float a = PI * (k * x - 1.0);
    return sin(a)/a;

}

void main() {
 
vec3 color = vec3(1.);
float d;

vec2 uv = gl_FragCoord.xy / resolution.xy - vec2(.5); 
uv.x *= resolution.x / resolution.y; 

d = length(uv - vec2(.5,0.));      

//color *= 1. - dot(uv,uv);
FragColor = vec4(color * sin(d * resolution.y) ,1.0);


}
