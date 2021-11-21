#version 330     
out vec4 fragColor;
uniform vec2 resolution;
uniform float time;
#define fragCoord gl_FragCoord

//two halves
//2021
//do

//#define main() mainImage(out vec4 fragColor,in vec2 fragCoord)
#define R resolution
#define t time

#define AA 2

#define EPS 0.0001
#define STEPS 245
#define NEAR 0.
#define FAR 3.

const int seed = 1290;

const float pi = radians(180.);
const float phi =  (1.+sqrt(5.))/2.;

float h11(float p) {
    uvec2 n = uint(int(p)) * uvec2(1391674541U,seed);
    uint h = (n.x ^ n.y) * 1391674541U;
    return float(h) * (1./float(0xffffffffU));
}

float h21(vec2 p) {
    uvec2 n = uvec2(ivec2(p))*uvec2(1391674541U,seed);
    uint h = (n.x^n.y) * 1391674541U;
    return float(h) * (1./float(0xffffffffU));
}

vec2 h22(vec2 p) {
    uvec2 n = uvec2(ivec2(p)) * uvec2(1391674541U,seed);
    n = (n.x ^ n.y) * uvec2(1391674541U,seed);
    return vec2(n) * (1./float(0xffffffffU));
}

float spiral(vec2 p,float s) {
    float d = length(p);
    float a = atan(p.x,p.y);
    float l = log(d)/.618 +a;
    return sin(l*s);
}

vec3 fmCol(float t,vec3 a,vec3 b,vec3 c,vec3 d) {
    return a + b * cos((radians(180.)*2.0) * (c * t + d));
}

vec2 opu(vec2 d1,vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
} 

float extr(vec3 p,float d,float h) {
    vec2 w = vec2(d,abs(p.z) - h);
    return min(max(w.x,w.y),0.) + length(max(w,0.)); 
} 

float n2(vec2 x) {
    vec2 p = floor(x);
    vec2 f = fract(x);
    f =  f*f*(3.-2.*f);
    float n = p.x + p.y * 57.;
    
    return( 
        mix(
        mix(h11(n+0.),h11(n+1.),f.x),
        mix(h11(n+57.),h11(n+58.),f.x),
        f.y)
        );
       
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

float f2(vec2 x) {

    float t = 0.0;

    float g = exp2(-.626); 

    float a = 0.5;
    float f = 1.0;

    for(int i = 0; i < 5; i++) {
    t += a * n2(f * x); 
    f *= 2.0; 
    a *=  g;  
    
    }    

    return t;
}

float f3 (vec3 p) {
    float f = 1.;
    mat3 m = mat3(vec2(.8,.6),h11(150.),
                  vec2(-.6,.8),h11(125.),
                  vec2(-.8,.6),h11(100.));

    f += .5    * n3(p); p = m*p*2.01;
    f += .25   * n3(p); p = m*p*2.04;
    f += .125  * n3(p); p = m*p*2.1;
    f += .0625 * n3(p);
    return f / .94;
}

float dd(vec3 p) {
    vec3 q = vec3(f3(p+vec3(0.,1.,2.)),
                  f3(p+vec3(4.,2.,3.)),
                  f3(p+vec3(2.,5.,6.)));
    vec3 r = vec3(f3(p + 4. * q + vec3(4.5,2.4,5.5)),
                  f3(p + 4. * q + vec3(2.25,5.,2.)),
                  f3(p + 4. * q + vec3(3.5,1.5,6.)));
    return f3(p + 4. * r);
}

mat2 rot(float a) {

    float c = cos(a);
    float s = sin(a);
    
    return mat2(c,-s,s,c);
}
 
vec3 raydir(vec2 uv,vec3 ro,vec3 ta,float fov) {

     vec3 f = normalize(ta - ro);
     vec3 r = normalize(cross(vec3(0.0,1.0,0.0),f));
     vec3 u = normalize(cross(f,r));

     vec3 d = normalize(uv.x * r
     + uv.y * u + f * fov);  

     return d;
}

vec2 scene(vec3 p) { 

vec2 res = vec2(1.0,0.0);
vec3 q = p;

float r = 1./phi;
float f = .005,h = .005;

float d = spiral(p.xz,1.)+r,g; 
res = opu(res,vec2(
    extr(p.xzy,d*f,h),0.));

g = spiral(-q.xy,1.)+r;
res = opu(res,vec2(
    extr(q,g*f,h),1.));





return res;

}

vec2 trace(vec3 ro,vec3 rd) {
    
    float d = -1.0;
    float s = NEAR;
    float e = FAR; 

    for(int i = 0; i < STEPS; i++) {

        vec3 p = ro + s * rd;
        vec2 dist = scene(p);
   
        if(abs(dist.x) < EPS || e <  dist.x ) { break; }
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
    
    for(int i = 0; i < 64; i++ ) {
        
        float h = scene(ro + rd * t  ).x;

        float y = h * h / (2. * ph);
        float d = sqrt(h*h-y*y);         
        res = min(res,212.*d/max(0.,t-y));
        ph = h;
        t += h;
    
        if(res < .001 || t > 23.) { break; }

        }

        return clamp(res,0.0,1.0);

}

vec3 calcNormal(vec3 p) {

    vec2 e = vec2(1.0,-1.0) * EPS;

    return normalize(vec3(
    vec3(e.x,e.y,e.y) * scene(p + vec3(e.x,e.y,e.y)).x +
    vec3(e.y,e.x,e.y) * scene(p + vec3(e.y,e.x,e.y)).x +
    vec3(e.y,e.y,e.x) * scene(p + vec3(e.y,e.y,e.x)).x + 
    vec3(    e.x    ) * scene(p + vec3(    e.x    )).x

    ));
    
}

vec3 render(vec3 ro,vec3 rd) {

    vec2 d = trace(ro,rd); 

    vec3 p = ro + rd * d.x;
    vec3 n = calcNormal(p);
    vec3 r = reflect(rd,n);    

    vec3 linear = vec3(0.);

    float amb = sqrt(clamp(.5+.5*n.y,0.,1.));
    float fre = pow(clamp(1.+dot(n,rd),0.,1.),2.);
    float ref = smoothstep(-2.,2.,r.y);

    vec3 col = vec3(.5); col = vec3(1.);
    vec3 bg_col = vec3(1.);
    col = bg_col * max(1.,rd.y);

    vec3 l = normalize(vec3(-25.,10.,5.));

    vec3 h = normalize(l - rd);  
    float dif = clamp(dot(n,l),0.0,1.0);
    float spe = pow(clamp(dot(n,h),0.0,1.0),16.) * dif * 
    (.04 + 0.9 * pow(clamp(1. + dot(h,rd),0.,1.),5.));

    if(d.y >= 0.) {

        dif *= shadow(p,l);
        ref  *= shadow(p,r);

        linear += dif * vec3(.5);
        linear += amb * vec3(0.001);
        linear += fre * vec3(.005,.02,.001);
        linear += spe * vec3(0.01,0.001,.005)*ref;

        if(d.y == 1.) {

            p *= 3.25;

            col += fmCol(dd(p),vec3(f3(p),h11(45.),h11(124.)),
                   vec3(h11(235.),f3(p),h11(46.)),
                   vec3(h11(245.),h11(75.),f3(p)),
                   vec3(1.,.5,.5));
        } else { 
            col = vec3(1.,0.,0.);
        }
        
          col = col * linear;
          col = mix(col,bg_col,1.-exp(-.0001*d.x*d.x*d.x)); 

}

return col;
}

void main() { 

vec3 color = vec3(0.);

vec3 ta = vec3(0.);
vec3 ro = vec3(.5);
ro.xz *= rot(t*.05);

for(int k = 0; k < AA; ++k) {
    for(int l = 0; l < AA; ++l) { 

vec2 o = vec2(float(l),float(k))/ float(AA) -.5;
vec2 uv = (2.* (fragCoord.xy+o) - R.xy)/R.y;

vec3 rd = raydir(uv,ro,ta,2.);
vec3 col = render(ro,rd);       

col = pow(col,vec3(.4545));
color += col;
   
    }
}

color /= float(AA*AA);
fragColor = vec4(color,1.0);

}
