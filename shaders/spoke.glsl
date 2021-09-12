#version 330     

// dolson
out vec4 FragColor; 

uniform vec2 resolution;

uniform float time;
uniform int frame; 
uniform float hour;
uniform float minute;
uniform float second;

uniform vec3 camPos;
uniform sampler2D tex;
uniform int seed;

#define AA 1
#define EPS 0.00001

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

vec3 h33(vec3 p) {
   uvec3 h = uvec3(ivec3(  p)) * 
   uvec3(1391674541U,seed,2860486313U);
   h = (h.x ^ h.y ^ h.z) * uvec3(1391674541U,seed,2860486313U);
   return vec3(h) * (1.0/float(0xffffffffU));

}

vec2 h12rad(float n) {
    float a = fract(sin(n*5673.)*48.)*radians(180.);
    float b = fract(sin(a+n)*6446.)*float(seed);
    return vec2(cos(a),sin(b));
}

float expStep(float x,float k) {
    return exp((x*k)-k);
}

float spiral(vec2 p,float n,float h) {
     float ph = pow(length(p),1./n)*32.;
     p *= mat2(cos(ph),sin(ph),sin(ph),-cos(ph));
     return h-length(p) / 
     sin((atan(p.x,-p.y)+radians(180.)/(radians(180.)*2.))*radians(180.)); 
}

float concentric(vec2 p,float h) {
    return cos(length(p))-h;
}

vec2 diag(vec2 uv) {
   vec2 r = vec2(0.);
   r.x = 1.1547 * uv.x;
   r.y = uv.y + .5 * r.x;
   return r;
}

vec3 simplexGrid(vec2 uv) {

    vec3 q = vec3(0.);
    vec2 p = fract(diag(uv));
    
    if(p.x > p.y) {
        q.xy = 1. - vec2(p.x,p.y-p.x);
        q.z = p.y;
    } else {
        q.yz = 1. - vec2(p.x-p.y,p.y);
        q.x = p.x;
    }
    return q;

}

float easeOut4(float t) {
    return -1.0 * t * (t - 2.0);
}

vec3 fmCol(float t,vec3 a,vec3 b,vec3 c,vec3 d) {
    return a + b * cos((radians(180.)*2.0) * (c * t + d));
}

vec3 contrast(vec3 c) {
return smoothstep(0.,1.,c);
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

vec2 sfold(vec2 p) {
    vec2 v = normalize(vec2(1.,-1.));
    float g = dot(p,v);
    return p-(g-sqrt(p*p+.01))*v;
}


vec2 radmod(vec2 p,float r) {
    float n = radians(360.)/r;
    float a = atan(p.x,p.y)+n*.5;
    a = floor(a/n)*n;
    return p * mat2(cos(a),-sin(a),sin(a),cos(a));
}

vec3 twist(vec3 p,float k) {
    
    float s = sin(k * p.y);
    float c = cos(k * p.y);
    mat2 m = mat2(c,-s,s,c);
    return vec3(m * p.xz,p.y);
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

float sphere(vec3 p,float r) {     
    return length(p) - r;
}

float cylinder(vec3 p,vec3 c) {
    return length(p.xz-c.xy)-c.z;
}

vec2 scene(vec3 p) { 

vec2 res = vec2(1.0,0.0);

vec3 q = p;

p.xz *= rot(time*.1);
p.xy = radmod(p.xy,12.);
p.y -= 4.;

float c = cylinder(p,vec3(0.,0.,.5));
float s = sphere(q,3.);

res = opu(res,vec2(smod(s,c,.05),12.));

return res;

}

vec2 trace(vec3 ro,vec3 rd) {
    
    float d = -1.0;
    float s = 0.;
    float e = 125.; 

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

    vec3 l = normalize(vec3(-100.));
    
    float rad = dot(rd,l);
    col += col * vec3(.5,.12,.25) * expStep(rad,100.);
    col += col * vec3(.5,.1,.15) * expStep(rad,25.);
    col += col * vec3(.1,.5,.05) * expStep(rad,2.);
    col += col * vec3(.15) * expStep(rad,35.);


    vec3 h = normalize(l - rd);  
    float dif = clamp(dot(n,l),0.0,1.0);
    float spe = pow(clamp(dot(n,h),0.0,1.0),16.)
    * dif * (.04 + 0.9 * pow(clamp(1. + dot(h,rd),0.,1.),5.));

    if(d.y >= 0.) {

        col = .2+.2*sin(2.*d.y+vec3(2.,3.,4.)); 

        dif *= shadow(p,l);
        ref *= shadow(p,r);

        linear += dif * vec3(1.);
        linear += amb * vec3(0.5);
        linear += fre * vec3(.025,.01,.03);
        linear += .25 * spe * vec3(0.04,0.05,.05)*ref;

        if(d.y == 12.) {

            p *= .25;

            col += fmCol(h11(164.),vec3(f3(p),h11(45.),h11(124.)),
                   vec3(h11(235.),f3(p),h11(46.)),
                   vec3(h11(245.),h11(75.),f3(p)),
                   vec3(1.));

            ref = vec3(0.005);     

        }    
 
        ro = p+n*.001*2.5;
        rd = r;

        col = col * linear;
        col = mix(col,vec3(.5),1.-exp(-.000001*d.x*d.x*d.x));

 }

return col;
}

void main() { 
vec3 color = vec3(0.);

vec3 ta = vec3(0.);
vec3 ro = vec3(0.,1.,5.);

for(int k = 0; k < AA; k++ ) {
   for(int l = 0; l < AA; l++) {
   
       vec2 o = vec2(float(k),float(l)) / float(AA) * .5;
       vec2 uv = (2.* (gl_FragCoord.xy+o) 
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
   }
}
   
   color /= float(AA*AA);
   FragColor = vec4(color,1.0);
 

}
