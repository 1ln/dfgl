#version 430 core     

// dolson,2019

out vec4 FragColor;

uniform sampler2D tex;

uniform vec2 resolution;

uniform float time;

uniform vec3 camPos;  
uniform vec3 camTar;

uniform int seed;
uniform int aa;

const int steps = 100;
const float eps = 0.0001;
const float dmin = 0.0;
const float dmax = 225.0;

const float fov = 2.;

const float PI   =  radians(180.0); 
const float PI2  =  PI * 2.;

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

float f2(vec2 x,int octaves) {

    float f = 0.;

    for(int i = 1; i < octaves; i++) {
 
    float e = pow(2.,float(i));
    float s = (1./e);
    f += ns2(x*e)*s;   
    
    }    

    return f * .5 + .5;
}

float sin2(vec2 p,float s) {
    
    return sin(p.x*s) * sin(p.y*s);
}

float envSt(float x,float k,float n) {

    return exp(-k * pow(x,n));
}

vec2 opu(vec2 d1,vec2 d2) {

    return (d1.x < d2.x) ? d1 : d2;
} 

float plane(vec3 p,vec4 n) {

    return dot(p,n.xyz) + n.w;
}

float hyperbola(vec3 p) { 

vec2 l = vec2(length(p.xz) ,-p.y);
float a = 0.5;
float d = sqrt((p.x+p.y)*(p.x+p.y)- 4. *(p.x*p.y-a)) + 0.5; 
return (-p.x-p.y+d)/2.0;

}

vec2 scene(vec3 p) {

    vec2 res = vec2(1.,0.);

    float d = 0.;     
    float s = 0.1;
    float t = time;  
    
    vec3 q = p;

    d = hyperbola(p);
     
    res = opu(res,vec2(d,2.)); 
  
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

float shadow(vec3 ro,vec3 rd ) {

    float res = 1.0;
    float t = 0.005;
    float ph = 1e10;
    
    for(int i = 0; i < 45; i++ ) {
        
        float h = scene(ro + rd * t  ).x;

        float y = h * h / (2. * ph);
        float d = sqrt(h*h-y*y);         
        res = min(res,25.0 * d/max(0.,t-y));
        ph = h;
        t += h;
    
        if(res < eps || t > 75.0) { break; }

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

vec3 renderNormals(vec3 ro,vec3 rd) {

   vec2 d = rayScene(ro,rd);
   vec3 p = ro + rd * d.x;
   vec3 n = calcNormal(p);   
   vec3 col = vec3(n);
   
   return col;
}

vec3 render(vec3 ro,vec3 rd) {
 
vec2 d = rayScene(ro, rd);

vec3 col = vec3(1.) - max(rd.y,0.);

if(d.y >= 0.) { 

vec3 p = ro + rd * d.x;
vec3 n = calcNormal(p);
vec3 l = normalize(vec3(0.,10.,10.));
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
ref *= shadow(p,r);

linear += dif * vec3(.5);
linear += amb * vec3(.05);
linear += ref * vec3(4.);
linear += fre * vec3(.25);

if(d.y == 2.) {

    float nl;
    nl = f3(p+f3(p,5,.5),6,.5);
    col += vec3(0.5,nl,dif.y);     
               
}

col = col * linear;

col += 5. * spe * vec3(.5);


} 

return col;
}

void main() {
 
vec3 color = vec3(0.);

for(int k = 0; k < aa; ++k) {
    for(int l = 0; l < aa; ++l) {

    vec2 o = vec2(float(l),float(k)) / float(aa) - .5;

    vec2 uv = -1. + 2. * (gl_FragCoord.xy + o) / resolution.xy; 
    uv.x *= resolution.x/resolution.y; 

    vec3 dir = rayCamDir(uv,camPos,camTar,fov); 
    vec3 col = render(camPos,dir);  
    color += col;
    }

color /= float(aa*aa);
color = pow(color,vec3(.4545));

FragColor = vec4(color,1.0);
}

}
