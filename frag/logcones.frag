#version 430 core

//dan olson
//2020

out vec4 FragColor;

uniform vec2 resolution;
uniform float time;

const int steps = 250;
const float eps = 0.0001;
const float trace_dist = 500.;

const float PI  =  radians(180.0); 
const float PI2 = PI * 2.;

vec3 fcos(vec3 x) { 
vec3 w = fwidth(x);
return cos(x) * smoothstep(PI2,0.,w);
}

float hash(float p) {

    uvec2 n = uint(int(p)) * uvec2(1391674541U,2531151992.0 * 5535123.);
    uint h = (n.x ^ n.y) * 1391674541U;
    return float(h) * (1.0/float(0xffffffffU));

}

float ns(vec3 x) {

    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 157.0 + 113.0 * p.z;

    return mix(mix(mix(hash(  n +   0.0) , hash(   n +   1.0)  ,f.x),
                   mix(hash(  n + 157.0) , hash(   n + 158.0)   ,f.x),f.y),
               mix(mix(hash(  n + 113.0) , hash(   n + 114.0)   ,f.x),
                   mix(hash(  n + 270.0) , hash(   n + 271.0)   ,f.x),f.y),f.z);
}

float f(vec3 x,int octaves,float h) {

    float t = 0.0;

    float g = exp2(-h); 

    float a = 0.5;
    float f = 1.0;

    for(int i = 0; i < octaves; i++) {
 
    t += a * ns(f * x); 
    f *= 2.0; 
    a *=  g;  
    
    }    

    return t;
}

float sin3(vec3 p,float h) {   
    return sin(p.x*h) * sin(p.y*h) * sin(p.z*h);
}

vec3 fmCol(float t,vec3 a,vec3 b,vec3 c,vec3 d) {
    return a + b * fcos((2. * PI) * (c * t + d));
}

vec2 opu(vec2 d1,vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
} 

float roundCone(vec3 p,float r1,float r2,float h) {

    vec2 q = vec2(length(vec2(p.x,p.y)),p.z);
    float b = (r1-r2)/h;
    float a = sqrt(1.0 - b*b);
    float k = dot(q,vec2(-b,a));

    if( k < 0.0) return length(q) - r1;
    if( k > a*h) return length(q - vec2(0.0,h)) - r2;

    return dot(q,vec2(a,b)) - r1;
} 

vec2 scene(vec3 p) { 

vec2 res = vec2(1.0,0.0);
float scale = float(45.) / PI;

vec2 h = p.xz; 
float r = length(h); 
h = vec2(log(r),atan(h.y,h.x));
h *= scale;
h = mod(h,2.) - 1.;
float mul = r/scale;

p.y += ns(p * .5) + .5;

float d = 0.;
d = roundCone(vec3(h,p.y/mul),1.,.5,2.) * mul; 

res = vec2(d,1.);

return res;

}

vec2 rayScene(vec3 ro,vec3 rd) {
    
    float depth = 0.0;
    float d = -1.0;

    for(int i = 0; i < steps; i++) {

        vec3 p = ro + depth * rd;
        vec2 dist = scene(p);
   
        if(abs( dist.x) < eps || trace_dist <  dist.x ) { break; }
        depth += dist.x;
        d = dist.y;

        }
 
        if(trace_dist < depth) { d = -1.0; }
        return vec2(depth,d);

}

float shadow(vec3 ro,vec3 rd ) {

    float res = 1.0;
    float t = 0.005;
    float ph = 1e10;
    
    for(int i = 0; i < 45; i++ ) {
        
        float h = scene(ro + rd * t  ).x;

        float y = h * h / (2. * ph);
        float d = sqrt(h*h-y*y);         
        res = min(res,100. * d/max(0.,t-y));
        ph = h;
        t += h;
    
        if(res < eps || t > 25.) { break; }

        }

        return clamp(res,0.0,1.0);

}

vec3 calcNormal(vec3 p) {

    vec2 e = vec2(1.0,-1.0) * eps;

    return normalize(vec3(
    vec3(e.x,e.y,e.y) * scene(p + vec3(e.x,e.y,e.y)).x +
    vec3(e.y,e.x,e.y) * scene(p + vec3(e.y,e.x,e.y)).x +
    vec3(e.y,e.y,e.x) * scene(p + vec3(e.y,e.y,e.x)).x + 
    vec3(e.x,e.x,e.x) * scene(p + vec3(e.x,e.x,e.x)).x

    ));

}

vec3 rayCamDir(vec2 uv,vec3 camPosition,vec3 camTarget,float fPersp) {

     vec3 camForward = normalize(camTarget - camPosition);
     vec3 camRight = normalize(cross(vec3(0.0,1.0,0.0),camForward));
     vec3 camUp = normalize(cross(camForward,camRight));


     vec3 vDir = normalize(uv.x * camRight + uv.y * camUp + camForward * fPersp);  

     return vDir;
}

vec3 render(vec3 ro,vec3 rd) {

vec2 d = rayScene(ro, rd);

vec3 col = vec3(1.);

if(d.y >= 0.) {

vec3 p = ro + rd * d.x;
vec3 n = calcNormal(p);
vec3 l = normalize(vec3(2.,10.,15.));
vec3 h = normalize(l - rd);
vec3 r = reflect(rd,n);

float amb = sqrt(clamp(0.5 + 0.5 * n.y,0.0,1.0));
float dif = clamp(dot(n,l),0.0,1.0);
float spe = pow(clamp(dot(n,h),0.0,1.0),16.) * dif * (.04 + 0.9 * pow(clamp(1. + dot(h,rd),0.,1.),5.));
float fre = pow(clamp(1. + dot(n,rd),0.0,1.0),2.0);
float ref = smoothstep(-.2,.2,r.y);
vec3 linear = vec3(0.1);

dif *= shadow(p,l);

linear += dif * vec3(.15);
linear += amb * vec3(.03);
linear += ref * vec3(.1);
linear += fre * vec3(.045); 

float n0,n1,n2,n3;

n0 = f(p,6,hash(100.));
n1 = f(p + f(p,6,hash(126.)),5,hash(10.));
n2 = sin3(p,f(p,8,hash(12.)));
n3 = f(p + sin3(p,hash(35.)),5,hash(132.));

col = fmCol(p.y,vec3(n1,hash(15.),hash(44.)),
               vec3(hash(112.),n2,hash(105.)),
               vec3(hash(62.),hash(201.),n3),
               vec3(1.));

col = col * linear;
col += 5. * spe * vec3(.5);
}
return col;
}

void main() {

vec3 cam_target = vec3(0.);
vec3 cam_pos = vec3(10.,25.,45.);

vec2 uv = -1. + 2. * gl_FragCoord.xy / resolution.xy;
uv.x *= resolution.x / resolution.y; 

vec3 dir = rayCamDir(uv,cam_pos,cam_target,5.);
vec3 color = render(cam_pos,dir);
      
color = pow(color,vec3(.4545));      
FragColor = vec4(color,1.0);
 
}
