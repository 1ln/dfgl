#version 330 core

// dolson,2019

out vec4 FragColor;

uniform vec2 resolution;
uniform float time;

float seed = 1525267.;

const int steps = 145;
float eps = 0.0001;
float dmin = 0.;
float dmax = 250.;

const int shsteps = 125;
float shmax = 5.;
float shblur = 45.;  

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
 
float sin2(vec2 p,float h) {
    
    return sin(p.x*h) * sin(p.y*h);
}

mat4 rotAxis(vec3 axis,float theta) {

axis = normalize(axis);

    float c = cos(theta);
    float s = sin(theta);

    float oc = 1.0 - c;

    return mat4(
 
        oc * axis.x * axis.x + c, 
        oc * axis.x * axis.y - axis.z * s,
        oc * axis.z * axis.x + axis.y * s, 
        0.0,
        oc * axis.x * axis.y + axis.z * s,
        oc * axis.y * axis.y + c, 
        oc * axis.y * axis.z - axis.x * s,
        0.0,
        oc * axis.z * axis.x - axis.y * s,
        oc * axis.y * axis.z + axis.x * s, 
        oc * axis.z * axis.z + c, 
        0.0,
        0.0,0.0,0.0,1.0);

}

vec2 opu(vec2 d1,vec2 d2) {

    return (d1.x < d2.x) ? d1 : d2;
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

float h = ns2(loc.xz*.25 + ns2(loc.xz*.005));

if(h < hash(25.)) {
   d = sphere(q,2.);

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


     vec3 vDir = normalize(uv.x * camRight + uv.y *
                 camUp + camForward * fPersp);  

     return vDir;
}

vec3 render(vec3 ro,vec3 rd) {
 
vec2 d = rayScene(ro, rd);

vec3 col = vec3(1.) - max(rd.y,0.);

if(d.y >= 0.) {

vec3 p = ro + rd * d.x;
vec3 n = calcNormal(p);

vec3 l = normalize(vec3(10.,15.,15.));
mat4 ra = rotAxis(vec3(1.,0.,0.),(radians(180)*2.0)*time*.0001);
l = (vec4(l,1.)*ra).xyz;

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

if(d.y == 1.) {
col += vec3(.5);
}

col = col * linear;
col += 5. * spe * vec3(1.,.5,.35);

col = fog(col,vec3(1.),.000025,d.x,3.);

}

return col;
}

void main() {
 
vec3 color = vec3(0.);

vec3 cam_tar = vec3(0.);
vec3 cam_pos = vec3(5.,12,5.);

cam_tar.z += time * .1;
cam_pos += cam_tar;

vec2 uv = -1. + 2. * gl_FragCoord.xy / resolution.xy; 
uv.x *= resolution.x / resolution.y; 

vec3 dir = rayCamDir(uv,cam_pos,cam_tar,1.);

color = render(cam_pos,dir);  
color = pow(color,vec3(.4545));      
FragColor = vec4(color,1.0);

}
