#version 330 core     

//Dan Olson
//2020

out vec4 FragColor;

uniform vec2 resolution;
uniform float time;
uniform int seed;

const int steps = 100;
float eps = 0.0001;
float dmin = 1.;
float dmax = 255.;
const int aa = 2;
 
const int shsteps = 45; 
float shblur = 10.0;
float shmax = 100.; 

const int octaves = 5;
float hurst = 0.5;

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
    float h = exp2(-hurst);     
    float f = 1.;
    float a = 0.5;

    for(int i = 1; i < octaves; i++) {
 
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

vec3 fmCol(float t,vec3 a,vec3 b,vec3 c,vec3 d) {
    return a + b * cos((radians(180) * 2.0) * (c * t + d));
}

mat2 rot2(float a) {

    float c = cos(a);
    float s = sin(a);
    
    return mat2(c,-s,s,c);
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

float plane(vec3 p,vec4 n) {
    return dot(p,n.xyz) + n.w;
}

float roundbox(vec3 p,vec3 b,float r) {

    vec3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
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

    float d = 0.;
    float b = 0.;     
    float t = time;  

    vec3 q = p;
    vec3 e = p;

    p.xz *= rot2(t);
    p.zy *= rot2(t);

    e.yz *= rot2(t);

    b = roundbox(e,vec3(1.),0.5);  
    d = trefoil(p,vec2(1.5,.25),3.,.25,.5); 
    b = smod(d,b,0.5); 

    d = smod(plane(q*.5,vec4(1.,-1.,-1.,0.)),d,0.5);
    res = opu(res,vec2(d,h11(135.))); 
    res = opu(res,vec2(b,h11(100.)*100.));

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

vec3 renderScene(vec3 ro,vec3 rd,vec3 lp,vec3 c) {
 
vec2 d = rayScene(ro, rd);

vec3 col = c * max(0.,rd.y);

if(d.y >= 0.) {

vec3 p = ro + rd * d.x;
vec3 n = calcNormal(p);
vec3 l = normalize(lp);
vec3 h = normalize(l - rd);
vec3 r = reflect(rd,n);

col = 0.2 + 0.2 * sin(2.*d.y + vec3(0.,4.,3.));

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
linear += ref * vec3(4.  );
linear += fre * vec3(0.25,0.5,0.35);

col = col * linear;
col += spe * vec3(1.,0.97,1.); 
col = mix(col,c,1.-exp(-0.001 * d.x*d.x*d.x)); 

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
  
    vec3 render = renderScene(ro,rd,vec3(10.),vec3(1.));
    color += render;
    }

color /= float(aa*aa);
color = pow(color,vec3(.4545));

FragColor = vec4(color,1.0);
}

}
