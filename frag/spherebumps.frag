#version 330 core

// dolson,2019

out vec4 FragColor;

uniform vec2 resolution;
uniform float time;
uniform int seed;

const float PI   =  radians(180.0); 
const float PI2  =  PI * 2.;

const int steps = 145;
float eps = 0.0001;
float dmin = 0.;
float dmax = 250.;

const int shsteps = 16;
float shmax = 5.;
float shblur = 10.;  

float h11(float p) {
    uvec2 n = uint(int(p)) * uvec2(uint(int(seed)),2531151992.0);
    uint h = (n.x ^ n.y) * uint(int(seed));
    return float(h) * (1./float(0xffffffffU));
}

float h21(vec2 p) {
    uvec2 n = uvec2(ivec2(p)) * uvec2(uint(int(seed)),2531151992.0);
    uint h = (n.x ^ n.y) * uint(int(seed));
    return float(h) * (1./float(0xffffffffU));
}

float n2(vec2 x) { 

    vec2 p = floor(x);
    vec2 f = fract(x);

    f = f * f * (3.0 - 2.0 * f);  
    float n = p.x + p.y * 57.;  

    return mix(mix(h11(n+0.),h11(n+1.),f.x),
               mix(h11(n+57.),h11(n+58.),f.x),f.y);  
}

float f2(vec2 x) {

    float s = 0.;
    float h = exp2(-0.45);     
    float f = 1.;
    float a = 0.5;

    for(int i = 1; i < 6; i++) {
 
        s += a * n2(f * x);
        f *= 2.;
        a *= h;
    }    

    return s;
}

float dd(vec2 p) {

   vec2 q = vec2(f2(p+vec2(3.,0.5)),
                 f2(p+vec2(1.,2.5)));

   vec2 r = vec2(f2(p + 4. * q + vec2(7.5,4.35)),
                 f2(p + 4. * q + vec2(5.6,2.2))); 

   return f2(p + 4. * r);
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

vec3 fmCol(float t,vec3 a,vec3 b,vec3 c,vec3 d) {
    
    return a + b * cos( (PI*2.0) * (c * t + d));
}

float easeInOut4(float t) {

    if((t *= 2.0) < 1.0) {
        return 0.5 * t * t;
    } else {
        return -0.5 * ((t - 1.0) * (t - 3.0) - 1.0);
    }
}

float easeOut3(float t) {

    return (t = t - 1.0) * t * t + 1.0;

}

float easeInOut3(float t) {

    if((t *= 2.0) < 1.0) {
        return 0.5 * t * t * t;
    } else { 
        return 0.5 * ((t -= 2.0) * t * t + 2.0);

    }
}

mat2 rot2(float a) {

    float c = cos(a);
    float s = sin(a);
    
    return mat2(c,-s,s,c);
}

vec2 opu(vec2 d1,vec2 d2) {

    return (d1.x < d2.x) ? d1 : d2;
} 

float opu(float d1,float d2) {
    
    return min(d1,d2);
}

float opi(float d1,float d2) {

    return max(d1,d2);
}

float opd(float d1,float d2) {

    return max(-d1,d2);
}

float smou(float d1,float d2,float k) {

    float h = clamp(0.5 + 0.5 * (d2-d1)/k,0.0,1.0);
    return mix(d2,d1,h) - k * h * (1.0 - h);
}

float smod(float d1,float d2,float k) {

    float h = clamp(0.5 - 0.5 * (d2+d1)/k,0.0,1.0);
    return mix(d2,-d1,h) + k * h * (1.0 - h);
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

vec3 h = vec3(h21(loc.xz),h11(loc.y),h21(loc.xz));

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

     vec3 vDir = normalize(uv.x * camRight + 
     uv.y * camUp + camForward * fPersp);  

     return vDir;
}

vec3 render(vec3 ro,vec3 rd) {
 
vec2 d = rayScene(ro, rd);

vec3 col = vec3(1.) - max(rd.y,0.);
col = vec3(0.);

if(d.y >= 0.) {

vec3 p = ro + rd * d.x;
vec3 n = calcNormal(p);
vec3 l = normalize(vec3(10.,15.,15.));

vec3 h = normalize(l - rd);
vec3 r = reflect(rd,n);

float amb = sqrt(clamp(0.5 + 0.5 * n.y,0.0,1.0));

float dif = clamp(dot(n,l),0.0,1.0);

float spe = pow(clamp(dot(n,h),0.0,1.0),16.) * dif * 
               (.04 + 0.9 * pow(clamp(1. + dot(h,rd),0.,1.),5.));

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

col = fog(col,vec3(1.),dd(p.xz) * 0.000025,d.x,3.);

}

return col;
}

void main() {
 
vec3 color = vec3(0.);

vec3 cam_tar = vec3(0.);
vec3 cam_pos = vec3(2.,5.,5.);

cam_tar.z += time * 10.;
cam_pos += cam_tar;

vec2 uv = -1. + 2. * gl_FragCoord.xy / resolution.xy; 
uv.x *= resolution.x / resolution.y; 

vec3 dir = rayCamDir(uv,cam_pos,cam_tar,1.);
color = render(cam_pos,dir);  
color = pow(color,vec3(.4545));      
FragColor = vec4(color,1.0);

}
