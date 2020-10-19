#version 430 core     

// dolson,2019

out vec4 FragColor;

uniform vec2 resolution;
uniform float time;
//uniform vec3 ro;  

const int seed = 51499145; 

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

    vec3 m = max(.5 - vec3(dot(x0,x0),dot(x12.xy,x12.xy),dot(x12.zw,x12.zw)),0.);
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

float f2(vec2 x) {

    float f = 0.;

    for(int i = 1; i < 6; i++) {
 
    float e = pow(2.,float(i));
    float s = (1./e);
    f += ns2(x*e)*s;   
    
    }    

    return f * .5 + .5;
}

float dd(vec2 p) {

vec2 q = vec2(f2(p + vec2(0.0,1.0)),
              f2(p + vec2(2.4,1.5)));

vec2 r = vec2(f2(p + 4.0 * q + vec2(5.4,4.8)),
              f2(p + 4.0 * q + vec2(6.8,9.1)));

return f2(p + 4.0 * r);
}

float hyperbola(vec3 p) { 

vec2 l = vec2(length(p.xz) ,-p.y);
float a = 0.5;
float d = sqrt((l.x+l.y)*(l.x+l.y)- 4. *(l.x*l.y-a)) + 0.5; 
return (-l.x-l.y+d)/2.0;

}

void main() {
 
    vec2 uv = -1. + 2. * gl_FragCoord.xy / resolution.xy; 
    uv.x *= resolution.x/resolution.y; 

    float fov = 1.0;
    float a = -0.75;
     
    vec3 p = vec3(0.0,-0.75,-1.0);

    vec3 d = vec3(uv*fov,1.);
    d.yz *= mat2(cos(a),sin(a),-sin(a),cos(a));

    for(int i = 0; i < 100; ++i) {
        p += d * hyperbola(p);        
    }

    float fd = 3.0;

    float fs = 1.0;
    float fa = 1.0;

    float n = dd(p.xz*0.12)+0.12;

    vec4 col = vec4(n * log((p.y+fd)*fs)/log((fd-fa)*fs));
    FragColor = vec4(col);

}
