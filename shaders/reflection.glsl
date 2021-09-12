#version 330     

// dolson
out vec4 FragColor; 

uniform vec2 resolution;
uniform float time;
uniform vec3 camPos;

uniform int seed;

#define EPS 0.0001
#define PI2 radians(180.)*2.

float h11(float p) {
    return fract(sin(p)*float(43758.5453+seed));
}

float h21(vec2 p) {
    return fract(sin(dot(p,vec2(12.9898,78.233)*float(43758.5453+seed))));
}

float checkerboard(vec3 p,float h) {
    vec3 q = floor(p*h);
    return mod(q.x+q.z,2.);
}

float concentric(vec2 p,float h) {
    return cos(length(p))-h;
}

vec2 julia(vec2 p,float n,float b,float f) {
    float k = 0.;
    for(int i = 0; i < 64; i++) {
    p = vec2(p.x*p.x-p.y*p.y,(p.x*p.y))-f;
    if(dot(p,p) > b) {
        break;
    }
    return p;
    }
}

vec3 fmCol(float t,vec3 a,vec3 b,vec3 c,vec3 d) {
    return a + b * cos((radians(180.)*2.0) * (c * t + d));
}

float expStep(float x,float k) {
    return exp((x*k)-k);
}

vec2 boxBound(vec3 ro,vec3 rd,vec3 rad) {
    vec3 m =  1./rd;
    vec3 n = m * ro;
    vec3 k = abs(m) * rad;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    return vec2(max(max(t1.x,t1.y),t1.z),
                min(min(t2.x,t2.y),t2.z));
}

vec2 opu(vec2 d1,vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
} 

float n3(vec3 x) {

    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 157.0 + 113.0 * p.z;

    return mix(mix(mix(h11(n + 0.0),h11(n + 1.0),f.x),
           mix(h11(n + 157.0),h11(n + 158.0),f.x),f.y),
           mix(mix(h11(n + 113.0),h11(n + 114.0),f.x),
           mix(h11(n + 270.0),h11(n + 271.0),f.x),f.y),f.z);
}

float f3(vec3 x) {

    float t = 0.0;

    float g = exp2(-.626); 

    float a = 0.5;
    float f = 1.0;

    for(int i = 0; i < 5; i++) {
    t += a * n3(f * x); 
    f *= 2.0; 
    a *=  g;  
    
    }    

    return t;
}

mat2 rot(float a) {

    float c = cos(a);
    float s = sin(a);
    
    return mat2(c,-s,s,c);
}

vec3 rayCamDir(vec2 uv,vec3 ro,vec3 ta,float fov) {

     vec3 f = normalize(ta - ro);
     vec3 r = normalize(cross(vec3(0.0,1.0,0.0),f));
     vec3 u = normalize(cross(f,r));

     vec3 d = normalize(uv.x * r
     + uv.y * u + f * fov);  

     return d;
}

float box(vec3 p,vec3 b) {

    vec3 d = abs(p) - b;
    return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0);
}

float dodecahedron(vec3 p,float r) {
vec4 v = vec4(0.,1.,-1.,0.5 + sqrt(1.25));
v /= length(v.zw);

float d;
d = abs(dot(p,v.xyw))-r;
d = max(d,abs(dot(p,v.ywx))-r);
d = max(d,abs(dot(p,v.wxy))-r);
d = max(d,abs(dot(p,v.xzw))-r);
d = max(d,abs(dot(p,v.zwx))-r);
d = max(d,abs(dot(p,v.wxz))-r);
return d;
}

float icosahedron(vec3 p,float r) {
    float d;
    d = abs(dot(p,vec3(.577)));
    d = max(d,abs(dot(p,vec3(-.577,.577,.577))));
    d = max(d,abs(dot(p,vec3(.577,-.577,.577))));
    d = max(d,abs(dot(p,vec3(.577,.577,-.577))));
    d = max(d,abs(dot(p,vec3(0.,.357,.934))));
    d = max(d,abs(dot(p,vec3(0.,-.357,.934))));
    d = max(d,abs(dot(p,vec3(.934,0.,.357))));
    d = max(d,abs(dot(p,vec3(-.934,0.,.357))));
    d = max(d,abs(dot(p,vec3(.357,.934,0.))));
    d = max(d,abs(dot(p,vec3(-.357,.934,0.))));
    return d-r;
}

vec2 scene(vec3 p) { 

vec2 res = vec2(1.0,0.0);

vec3 q = p;

p.xz *= rot(-PI2*h11(100.));
p.zy *= rot(PI2*h11(125.));

float d = 1.;
float b,b1,b2,b3,b4;

b = box(q-vec3(2.),vec3(1.));
b1 = box(q-vec3(2.,2.,4.),vec3(.5));
b2 = box(q-vec3(-2.,2.5,-4.),vec3(2.));
b3 = box(q-vec3(4.,2.,-2.),vec3(1.25));
b4 = box(q-vec3(-4.,2.,-3.),vec3(1.));

d = dodecahedron(p-vec3(0.,1.,0.),1.);

res = opu(res,vec2(b,2.)); 
res = opu(res,vec2(b1,16.));
res = opu(res,vec2(b2,58.));
res = opu(res,vec2(b3,12.));
res = opu(res,vec2(b4,71.));

res = opu(res,vec2(max(-q.y,d),1.));

return res;


}

vec2 trace(vec3 ro,vec3 rd) {
    
    float d = -1.0;
    float s = 0.;
    float e = 25.;  

    for(int i = 0; i < 255; i++) {

        vec3 p = ro + s * rd;
        vec2 dist = scene(p);
   
        if(abs(dist.x) < EPS || e <  dist.x ) { break; }
        s += dist.x;
        d = dist.y;

        }
 
        if(e < s) { d = -1.0; }
        return vec2(s,d);

}

float reflection(vec3 ro,vec3 rd ) {

    float depth = 0.;
    float dmax = 100.;
    float d = -1.0;

    for(int i = 0; i < 125; i++ ) {
        float h = scene(ro + rd * depth).x;

        if(h < EPS) { return depth; }
        
        depth += h;
    }

    if(dmax <= depth ) { return dmax; }
    return dmax;
}

float shadow(vec3 ro,vec3 rd ) {

    float res = 1.0;
    float t = 0.005;
    float ph = 1e10;
    
    for(int i = 0; i < 125; i++ ) {
        
        float h = scene(ro + rd * t  ).x;

        float y = h * h / (2. * ph);
        float d = sqrt(h*h-y*y);         
        res = min(res,100. * d/max(0.,t-y));
        ph = h;
        t += h;
    
        if(res < EPS || t > 12.) { break; }

        }

        return clamp(res,0.0,1.0);

}

vec3 calcNormal(vec3 p) {

    vec2 e = vec2(1.0,-1.0) * EPS;

    return normalize(vec3(
    vec3(e.x,e.y,e.y) * scene(p + vec3(e.x,e.y,e.y)).x +
    vec3(e.y,e.x,e.y) * scene(p + vec3(e.y,e.x,e.y)).x +
    vec3(e.y,e.y,e.x) * scene(p + vec3(e.y,e.y,e.x)).x + 
    vec3(e.x,e.x,e.x) * scene(p + vec3(e.x,e.x,e.x)).x

    ));
    
}

vec3 render(inout vec3 ro,inout vec3 rd,inout vec3 ref) {


    vec2 d = trace(ro, rd);
    vec3 p = ro + rd * d.x;
    vec3 n = calcNormal(p);
    vec3 linear = vec3(0.);
    vec3 r = reflect(rd,n); 
    float amb = sqrt(clamp(.5+.5*n.x,0.,1.));
    float fre = pow(clamp(1.+dot(n,rd),0.,1.),2.);
    vec3 col = vec3(.5);

    vec3 l = normalize(vec3(1e10,0.,1e10));

    float rad = dot(rd,l);
    col += col * vec3(.5,.12,.25) * expStep(rad,100.);
    col += col * vec3(.5,.1,.15) * expStep(rad,250.);
    col += col * vec3(.1,.5,.05) * expStep(rad,25.);
    col += col * vec3(.15) * expStep(rad,35.);

    vec3 h = normalize(l - rd);  
    float dif = clamp(dot(n,l),0.0,1.0);
    float spe = pow(clamp(dot(n,h),0.0,1.0),16.)
    * dif * (.04 + 0.9 * pow(clamp(1. + dot(h,rd),0.,1.),5.));

    if(d.y >= 0.) {

        dif *= shadow(p,l);
        ref *= shadow(p,r);

        linear += dif * vec3(1.);
        linear += amb * vec3(0.5);
        linear += fre * vec3(.025,.01,.03);
        linear += .25 * spe * vec3(0.04,0.05,.05)*ref;

        if(d.y == 2.) {
            col = vec3(1.,0.,0.);
            ref = vec3(0.12);   
        }    
  
        if(d.y == 16.) {
            col = vec3(.5);
            ref = vec3(.25);
        }

        if(d.y == 12.) {
            col = vec3(concentric(p.xy,24.));
        }

        if(d.y == 58.) {
            col = vec3(julia(p.xy,-h11(95.),4.,h11(292.)); 
        }

        if(d.y == 71.) {
            col = vec3(checkerboard(p,10.));
            ref = vec3(.5);  
        }
        
        ro = p+n*.001*2.5;
        rd = r;

        col = col * linear;
        col = mix(col,vec3(.5),1.-exp(-.0001*d.x*d.x*d.x));
    }

return col;
}

void main() { 
vec3 color = vec3(0.);

vec3 ta = vec3(0.);
vec3 ro = vec3(-2.,5.,3.);

       vec2 uv = (2.* (gl_FragCoord.xy) 
       - resolution.xy)/resolution.y;

       vec3 rd = rayCamDir(uv,ro,ta,1.); 
       vec3 ref = vec3(0.);
       vec3 col = render(ro,rd,ref);       
       vec3 dec = vec3(1.);

       for(int i = 0; i < 2; i++) {
           dec *= ref;
           col += dec * render(ro,rd,ref);
       }

    col = pow(col,vec3(.4545));
    color += col;
    FragColor = vec4(color,1.0);
 

}
