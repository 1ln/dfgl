#version 430 core

//dan olson
//2020

out vec4 FragColor;

uniform vec2 resolution;
uniform float time;

uniform vec3 camPos;
uniform vec3 camTar;

float seed = 1525267.;

const int steps = 145;
float eps = 0.00001;
float dmin = 0.;
float dmax = 250.;

const int shsteps = 100;
float shmax = 45.;
float shblur = 75.;  

float hash(float p) {
    return fract(sin(p) * seed);
}

float hash(vec2 p) {
    return fract(sin(dot(p.xy,vec2(12.9898,78.233)))*seed);
}

vec2 opu(vec2 d1,vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
} 

float smou(float d1,float d2,float k) {

    float h = clamp(0.5 + 0.5 * (d2-d1)/k,0.0,1.0);
    return mix(d2,d1,h) - k * h * (1.0 - h);
}

float sphere(vec3 p,float r) { 
    return length(p) - r;
}

float plane(vec3 p,vec4 n) {
    return dot(p,n.xyz) + n.w;
}

vec2 scene(vec3 p) { 

vec2 res = vec2(1.0,0.0);
float d = 0.;

vec3 q = vec3(p); 

float s = 5.;
vec3 loc = floor(p/s);
q.xz = mod(q.xz,s) - .5 * s;

vec3 h = vec3(hash(loc.xz),hash(loc.y),hash(loc.xz));

if(h.x < .45) {
   d = sphere(q,1.);
} else {
   d = 1.;
}

float pl = plane(p,vec4(0.,1.,0.,1.));

res = opu(res,vec2(smou(pl,d,.5),2.));

return res;

}

vec2 rayScene(vec3 ro,vec3 rd) {
    
    float d = -1.0;
    float s = dmin;
    float e = dmax;  

    for(int i = 0; i < steps; i++) {

        vec3 p = ro + s * rd;
        vec2 dist = scene(p);
   
        if(abs(dist.x) < eps || e <  dist.x ) { break; }
        s += dist.x;
        d = dist.y;

        }
 
        if(e < s) { d = -1.0; }
        return vec2(s,d);

}

vec3 fog(vec3 col,vec3 fcol,float fdist,float fdens,float y) {

    float fdep = 1. - exp(-fdist * pow(fdens,y));
    return mix(col,fcol,fdep);
}

float shadow(vec3 ro,vec3 rd ) {

    float res = 1.0;
    float t = 0.005;
    float ph = 1e10;
    
    for(int i = 0; i < shsteps; i++ ) {
        
        float h = scene(ro + rd * t  ).x;

        float y = h * h / (2. * ph);
        float d = sqrt(h*h-y*y);         
        res = min(res,shblur * d/max(0.,t-y));
        ph = h;
        t += h;
    
        if(res < eps || t > shmax) { break; }

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

     vec3 vDir = normalize(
     uv.x * camRight + uv.y * camUp + camForward * fPersp);  

     return vDir;
}

void main() {

vec2 uv = -1.0 + 2.0 * gl_FragCoord.xy / resolution.xy;
uv.x *= resolution.x / resolution.y;

vec3 ro = camPos;
vec3 rd = rayCamDir(uv,ro,camTar,1.);
vec2 d = rayScene(ro, rd);

vec3 col = vec3(0.);

if(d.y >= 0.) {

vec3 p = ro + rd * d.x;
vec3 n = calcNormal(p);

vec3 l = normalize(camPos - camTar);

vec3 h = normalize(l - rd);
vec3 r = reflect(rd,n);

float amb = sqrt(clamp(0.5 + 0.5 * n.y,0.0,1.0));
float dif = clamp(dot(n,l),0.0,1.0);
float spe = pow(clamp(dot(n,h),0.0,1.0),16.) * dif * (.04 + 0.9 * pow(clamp(1. + dot(h,rd),0.,1.),5.));
float fre = pow(clamp(1. + dot(n,rd),0.0,1.0),2.0);
float ref = smoothstep(-.2,.2,r.y);
vec3 linear = vec3(0.);

dif *= shadow(p,l);
ref *= shadow(p,r);

linear += dif * vec3(.5);
linear += amb * vec3(0.02);
linear += ref * vec3(.05);
linear += fre * vec3(.25);

if(d.y == 2.) {
col += vec3(.5);
}

col = col * linear;
col += 5. * spe * vec3(1.,.5,.35);

col = fog(col,vec3(1.),.000025,d.x,3.);

}

col = pow(col,vec3(.4545));      
FragColor = vec4(col,1.0);

}
