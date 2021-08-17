#version 430 core

// dolson,2019

out vec4 FragColor; 

uniform vec2 resolution;
uniform float time;

const float PI  =  radians(180.0); 
const float PHI =  (1.0 + sqrt(5.0)) / 2.0;

float hash(float p) {
return fract(sin(p) * 4358.5453);
}

float hash(vec2 p) {
return fract(sin(dot(p.xy,vec2(12.9898,78.233))) * 43758.5357); 
} 

float noise(vec3 x) {

    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 157.0 + 113.0 * p.z;

    return mix(mix(mix(hash(  n +   0.0) , hash(   n +   1.0)  ,f.x),
                   mix(hash(  n + 157.0) , hash(   n + 158.0)   ,f.x),f.y),
               mix(mix(hash(  n + 113.0) , hash(   n + 114.0)   ,f.x),
                   mix(hash(  n + 270.0) , hash(   n + 271.0)   ,f.x),f.y),f.z);
}

mat2 rot(float a) {

    float c = cos(a);
    float s = sin(a);
    
    return mat2(c,-s,s,c);
}

float checker(vec2 p) {

    vec2 w = fwidth(p)-0.001;
    vec2 i = 2.*(abs(fract((p-.5*w)*.5)-.5)
             - abs(fract((p+.5*w)*.5)-.5))/w;
    return 0.5 - 0.5 * i.x * i.y;
}

vec2 opu(vec2 d1,vec2 d2) {

    return (d1.x < d2.x) ? d1 : d2;
} 

float box(vec3 p,vec3 b) {

    vec3 d = abs(p) - b;
    return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0);
}

vec2 scene(vec3 p) { 

vec2 res = vec2(1.0,0.0);

float t = time;

vec3 q = p;

q.z += time*2.; 
q.xz *= sin(q.xz * .5 + noise(q + noise(q)) + t ) + .5;
q.z *= cos(q.z * .5 + noise(q) + t) + .5; 

float pl = dot(q*.5,vec3(.5,-1.,0.));

res = opu(res,vec2(max(-pl,box(p,vec3(1.))-.15),2.));
res = opu(res,vec2(p.y+1.,1.));

return res;

}

vec2 rayScene(vec3 ro,vec3 rd) {
    
    float depth = 0.0;
    float d = -1.0;

       for(int i = 0; i < 245; i++) {

        vec3 p = ro + depth * rd;
        vec2 dist = scene(p);
   
        if(abs( dist.x) < 0.001 || 150. <  dist.x ) { break; }


        depth += dist.x;
        d = dist.y;

        }
 
        if(150. < depth) { d = -1.0; }

        return vec2(depth,d);

}

float shadow(vec3 ro,vec3 rd ) {

    float res = 1.0;
    float t = 0.005;
    float ph = 1e10;
    
    for(int i = 0; i < 245; i++ ) {

        float h = scene(ro + rd * t  ).x;

        float y = h * h / (2. * ph);
        float d = sqrt(h*h-y*y);         
        res = min(res,125. * d/max(0.,t-y));
        ph = h;
        t += h;
    
        if(res < 0.00001 || t >  10. ) { break; }
        }

        return clamp(res,0.0,1.0);

}

vec3 calcNormal(vec3 p) {

    vec2 e = vec2(1.0,-1.0) * 0.0001;

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

     vec3 vDir = normalize(uv.x * camRight + uv.y
     * camUp + camForward * fPersp);  

     return vDir;
}

vec3 render(vec3 ro,vec3 rd) {

vec2 d = rayScene(ro, rd);

vec3 cf = vec3(1.);                         
vec3 col = cf - max(rd.y,0.);

if(d.y >= 0.) {

vec3 p = ro + rd * d.x;
vec3 n = calcNormal(p);
vec3 l = normalize( vec3(25.,45.,33.));
l.xz *= rot(time*.05);

vec3 h = normalize(l - rd);
vec3 r = reflect(rd,n);

float amb = sqrt(clamp(0.5 + 0.5 * n.y,0.0,1.0));
float dif = clamp(dot(n,l),0.0,1.0);

float spe = pow(clamp(dot(n,h),0.0,1.0),16.)
* dif * (.04 + 0.9 * pow(clamp(1. + dot(h,rd),0.,1.),5.));

float fre = pow(clamp(1. + dot(n,rd),0.0,1.0),2.0);
float ref = smoothstep(-.2,.2,r.y);

vec3 linear = vec3(0.);

dif *= shadow(p,l);
ref *= shadow(p,l);

linear += dif * vec3(.5);
linear += amb * vec3(.06,.05,.01); 
linear += ref * vec3(.0045,.0044,.004); 
linear += fre * vec3(.005,.0033,.001);

if(d.y == 2.) {
    col = vec3(checker(p.xz*12.) * .5 + .1);
}

if(d.y == 1.) {
    col = vec3(.5);
} 

col = col * linear;
col += 5. * spe * vec3(1.,.5,.5 );
}

return col;
}

void main() {
 
vec3 cam_target =  vec3(0.);
vec3 cam_pos = vec3(3.,5.,12.);
cam_pos.xz *= rot(time * .12);

vec2 uv =  -1. + 2. * gl_FragCoord.xy / resolution.xy;
uv.x *= resolution.x / resolution.y; 

vec3 dir = rayCamDir(uv,cam_pos,cam_target,5.);
vec3 color = render(cam_pos,dir);
color = pow(color,vec3(.4545));      
FragColor = vec4(color,1.0);

}
