#version 330 core     

//Dan Olson

out vec4 FragColor;

uniform vec2 resolution;
uniform float time;
const int seed = 153435;

const int steps = 100;
float eps = 0.0001;
float dmin = 0.;
float dmax = 255.;
const int aa = 2;
 
const int shsteps = 125; 
float shblur = 70.;
float shmax = 10.; 

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

float sin3(vec3 p,vec3 f) { 
    return sin(p.x) * f.x + sin(p.y) * f.y + sin(p.z) * f.z;
}
    
mat2 rot2(float a) {

    float c = cos(a);
    float s = sin(a);
    
    return mat2(c,-s,s,c);
}

vec3 twist(vec3 p,vec4 w) {
    float c = cos(w.x*p.y+w.y);
    float s = sin(w.z*p.y+w.w);
    mat2 m = mat2(c,-s,s,c);
    return vec3(p.xz*m,p.y);
}

mat3 camOrthographic(vec3 ro,vec3 ta,float r) {
     
     vec3 w = normalize(ta - ro); 
     vec3 p = vec3(sin(r),cos(r),0.);           
     vec3 u = normalize(cross(w,p)); 
     vec3 v = normalize(cross(u,w));

     return mat3(u,v,w); 
} 

vec2 opu(vec2 d1,vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
} 

float smod(float d1,float d2,float k) {

    float h = clamp(0.5 - 0.5 * (d2+d1)/k,0.0,1.0);
    return mix(d2,-d1,h) + k * h * (1.0 - h);
}

float smoi(float d1,float d2,float k) {

    float h = clamp(0.5 + 0.5 * (d2-d1)/k,0.0,1.0);
    return mix(d2,d1,h) + k * h * (1.0 - h);

}

float plane(vec3 p,vec4 n) {
    return dot(p,n.xyz) + n.w;
}

float trefoil(vec3 p,vec2 t,float n,float l,float e) {

    vec2 q = vec2(length(p.xz)-t.x,p.y);     

    float a = atan(p.x,p.z);
    float c = cos(a*n);
    float s = sin(a*n);

    mat2 m = mat2(c,-s,s,c);    
    q *= m;

    q.y = abs(q.y)-l;
 
    return (length(q) - t.y)*e;

}

vec2 scene(vec3 p) {

    vec2 res = vec2(1.,0.);    
    float t = time;  

    vec3 q = p;

    p.xz *= rot2(time*.1);
    p.zy *= rot2(time*.125);    
    
    vec4 t1 = vec4(.25,.1,h11(245.),-.25);
   
    t1 = mix(vec4(-.5,h11(125.),-.25,h11(111.)),vec4(0.),
    step(h11(12.),h11(235.)));

    t1 = mix(vec4(h11(100.),-.25,-h11(35.),.5),vec4(0.),
    step(h11(155.),h11(95.)));

    res = opu(res,vec2(
        trefoil(twist(p,t1),vec2(1.5,.25),3.,.35,.5),25.));  

    float pl = plane(q*.5,vec4(1.,1.,0.,0.));

    res = opu(res,vec2(smod(
    trefoil(twist(p,t1),vec2(1.5,.35),3.,.35,.5),pl,.1)
    ,12.5));

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

 
float shadow(vec3 ro,vec3 rd) {

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

vec3 renderScene(vec3 ro,vec3 rd) {
 
vec2 d = rayScene(ro, rd);

vec3 bg = vec3(.25,.15,.15);
vec3 col = bg * max(0.,rd.y);

if(d.y >= 0.) {

vec3 p = ro + rd * d.x;
vec3 n = calcNormal(p);
vec3 l = normalize(vec3(3.,5.,-6.));
vec3 h = normalize(l - rd);
vec3 r = reflect(rd,n);

if(d.y == 25.) {
    col = vec3(sin3(p,vec3(h11(222.),h11(155.),h11(335.))*.025));
} else {
    col = vec3(.9);
}

float amb = clamp(0.5 + 0.5 * n.y,0.,1.);

float dif = clamp(dot(n,l),0.0,1.0);

float spe = pow(clamp(dot(n,h),0.0,1.0),16.)
* dif * (.04 + 0.9 * pow(clamp(1. + dot(h,rd),0.,1.),5.));

float fre = pow(clamp(1. + dot(n,rd),0.0,1.0),2.0);
float ref = smoothstep(-.2,.2,r.y);

vec3 linear = vec3(0.);

dif *= shadow(p,l);
ref *= shadow(p,r);

linear += dif * vec3(.5); 
linear += amb * vec3(0.01,0.05,0.05);
linear += ref * vec3(0.05,0.01,0.05);
linear += fre * vec3(0.25,0.5,0.35);

col = col * linear;
col += spe * vec3(.25,.01,.05); 
col = mix(col,bg,1.-exp(-0.01 * d.x*d.x*d.x)); 

}

return col;
}

void main() {
 
vec3 color = vec3(0.);
vec3 ro = vec3(3.);
vec3 ta = vec3(0.0);

for(int k = 0; k < aa; ++k) {
    for(int l = 0; l < aa; ++l) {

    vec2 o = vec2(float(l),float(k)) / float(aa) - .5;

    vec2 uv = (2. * (gl_FragCoord.xy + o) -
    resolution.xy) / resolution.y; 

    mat3 cm = camOrthographic(ro,ta,0.);
    vec3 rd = cm * normalize(vec3(uv.xy,2.));
  
    vec3 render = renderScene(ro,rd);
    render = pow(render,vec3(.4545));
    color += render;

    }
}

color /= float(aa*aa);
FragColor = vec4(color,1.0);

}
