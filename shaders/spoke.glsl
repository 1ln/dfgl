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
#define EPS 0.0001 
#define HASH_TYPE

float dot2(vec2 v) { return dot(v,v); }
float dot2(vec3 v) { return dot(v,v); }
float ndot(vec2 a,vec2 b) { return a.x * b.x - a.y * b.y; }

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

float easeInOut4(float t) {
    if((t *= 2.0) < 1.0) {
        return 0.5 * t * t;
    } else {
        return -0.5 * ((t - 1.0) * (t - 3.0) - 1.0);
    }
}

float easeIn3(float t) {
    return t * t * t;
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

vec3 fmCol(float t,vec3 a,vec3 b,vec3 c,vec3 d) {
    return a + b * cos((radians(180.)*2.0) * (c * t + d));
}

vec3 contrast(vec3 c) {
return smoothstep(0.,1.,c);
}

vec3 repLim(vec3 p,float c,vec3 l) {
  
    vec3 q = p - c * clamp( floor((p/c)+0.5) ,-l,l);
    return q; 
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

vec4 el(vec3 p,vec3 h) {
    vec3 q = abs(p) - h;
    return vec4(max(q,0.),min(max(q.x,max(q.y,q.z)),0.));
}

float extr(vec3 p,float d,float h) {
    vec2 w = vec2(d,abs(p.z) - h);
    return min(max(w.x,w.y),0.) + length(max(w,0.)); 
} 

vec2 rev(vec3 p,float w,float f) {
    return vec2(length(p.xz) - w * f,p.y);
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

float cell(vec3 x,float n) {
 
    x *= n;
    vec3 p = floor(x);
    vec3 f = fract(x);
 
    float min_dist = 1.0;
    
    for(int i = -1; i <= 1; i++) {
        for(int j = -1; j <= 1; j++) {
            for(int k = -1; k <= 1; k++) { 

                vec3 b = vec3(float(k),float(j),float(i));
                vec3 r = h33( p + b );
                
                vec3 diff = (b + r - f);

                float d = length(diff);
                min_dist = min(min_dist,d);
    
            }
        }
    }
 
    return min_dist;  

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

float f4(vec3 p) {
    float f = 1.;
    mat3 m = mat3(vec2(.8,.6),h11(150.),
                  vec2(-.6,.8),h11(125.),
                  vec2(-.8,.6),h11(100.));

    f += .5    * sin3(p,1.); p = m*p*2.01;
    f += .25   * sin3(p,1.); p = m*p*2.04;
    f += .125  * sin3(p,1.); p = m*p*2.1;
    f += .0625 * sin3(p,1.);
    return f / .94;
}

mat2 rot(float a) {

    float c = cos(a);
    float s = sin(a);
    
    return mat2(c,-s,s,c);
}

mat4 rotAxis(vec3 axis,float theta) {

axis = normalize(axis);

    float c = cos(theta);
    float s = sin(theta);

    float oc = 1.0 - c;

    return mat4(
 
        oc * axis.x * axis.x + c, 
        oc * axis.x * axis.y - axis.z * s,
        oc * axis.z * axis.x + axis.y * s,0.,
    
        oc * axis.x * axis.y + axis.z * s,
        oc * axis.y * axis.y + c, 
        oc * axis.y * axis.z - axis.x * s,0.,

        oc * axis.z * axis.x - axis.y * s,
        oc * axis.y * axis.z + axis.x * s,
        oc * axis.z * axis.z + c,0.,0.,0.,0.,1.);

}


vec3 rayCamDir(vec2 uv,vec3 ro,vec3 ta,float fov) {

     vec3 f = normalize(ta - ro);
     vec3 r = normalize(cross(vec3(0.0,1.0,0.0),f));
     vec3 u = normalize(cross(f,r));

     vec3 d = normalize(uv.x * r
     + uv.y * u + f * fov);  

     return d;
}

mat3 camOrthographic(vec3 ro,vec3 ta,float r) {
     
     vec3 w = normalize(ta - ro); 
     vec3 p = vec3(sin(r),cos(r),0.);           
     vec3 u = normalize(cross(w,p)); 
     vec3 v = normalize(cross(u,w));

     return mat3(u,v,w); 
} 

float circle(vec2 p,float r) {
    return length(p) - r;
}

float ring(vec2 p,float r,float w) {
    return abs(length(p) - r) - w;
}

float eqTriangle(vec2 p,float r) { 

     const float k = sqrt(3.);
   
     p.x = abs(p.x) - 1.;
     p.y = p.y + 1./k;

     if(p.x + k * p.y > 0.) {
         p = vec2(p.x - k * p.y,-k * p.x - p.y)/2.;
     }

     p.x -= clamp(p.x,-2.,0.);
     return -length(p) * sign(p.y);    

}
 
float rect(vec2 p,vec2 b) {
    vec2 d = abs(p)-b;
    return length(max(d,0.)) + min(max(d.x,d.y),0.);
}

float roundRect(vec2 p,vec2 b,vec4 r) {
    r.xy = (p.x > 0.) ? r.xy : r.zw;
    r.x  = (p.y > 0.) ? r.x  : r.y;
    vec2 q = abs(p) - b + r.x;
    return min(max(q.x,q.y),0.) + length(max(q,0.)) - r.x;
}

float rhombus(vec2 p,vec2 b) {
    vec2 q = abs(p);
    float h = clamp(-2. * ndot(q,b)+ndot(b,b) / dot(b,b),-1.,1.);
    float d = length(q - .5 * b * vec2(1.- h,1. + h));
    return d * sign(q.x*b.y + q.y*b.x - b.x*b.y);  
}

float segment(vec2 p,vec2 a,vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa,ba)/dot(ba,ba),0.,1.);  
    return length(pa - ba * h);
}

float sphere(vec3 p,float r) { 
    
    return length(p) - r;
}

float ellipsoid(vec3 p,vec3 r) {

    float k0 = length(p/r); 
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}

float cone(vec3 p,vec2 c,float h) {
    vec2 q = h*vec2(c.x/c.y,-1.);
    vec2 w = vec2(length(p.xz),p.y);
    vec2 a = w -q * clamp(dot(w,q)/dot(q,q),0.,1.);
    vec2 b = w -q * vec2(clamp(w.x/q.x,0.,1.),1.);
    float k = sign(q.y);
    float d = min(dot(a,a),dot(b,b));
    float s = max(k*(w.x*q.y-w.y*q.x),k*(w.y-q.y));
    return sqrt(d)*sign(s);

}

float roundCone(vec3 p,float r1,float r2,float h) {

    vec2 q = vec2(length(vec2(p.x,p.z)),p.y);
    float b = (r1-r2)/h;
    float a = sqrt(1.0 - b*b);
    float k = dot(q,vec2(-b,a));

    if( k < 0.0) return length(q) - r1;
    if( k > a*h) return length(q - vec2(0.0,h)) - r2;

    return dot(q,vec2(a,b)) - r1;
}

float solidAngle(vec3 p,vec2 c,float ra) {
    
    vec2 q = vec2(length(vec2(p.x,p.z)),p.y);
    float l = length(q) - ra;
    float m = length(q - c * clamp(dot(q,c),0.0,ra));
    return max(l,m * sign(c.y * q.x - c.x * q.y));
} 

float prism(vec3 p,vec2 h) {

    vec3 q = abs(p);
    return max(q.z - h.y,  
    max(q.x * 0.866025 + p.y * 0.5,-p.y) - h.x * 0.5); 
}

float box(vec3 p,vec3 b) {

    vec3 d = abs(p) - b;
    return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0);
}

float torus(vec3 p,vec2 t) {

    vec2 q = vec2(length(vec2(p.x,p.z)) - t.x,p.y);
    return length(q) - t.y; 
}

float capTorus(vec3 p,vec2 sc,float ra,float rb) {
    p.x = abs(p.x);
    float k = (sc.y * p.x > sc.x * p.y) ? dot(p.xy,sc) : length(p.xy);
    return sqrt(dot(p,p) + ra*ra - 2.*k*ra) - rb;
}

float cylinder(vec3 p,vec3 c) {
    return length(p.xy-c.xz)-c.z;
}

float cylinder(vec3 p,float h,float r) {
    vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
    return min(max(d.x,d.y),0.) + length(max(d,0.));
}

float hexPrism(vec3 p,vec2 h) {
 
    const vec3 k = vec3(-0.8660254,0.5,0.57735);
    p = abs(p); 
    p.xy -= 2.0 * min(dot(k.xy,p.xy),0.0) * k.xy;
 
    vec2 d = vec2(length(p.xy 
           - vec2(clamp(p.x,-k.z * h.x,k.z * h.x),h.x))
           * sign(p.y-h.x),p.z-h.y);

    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float pyramid(vec3 p,float h) {
    float m2 = h*h + .25;
    p.xz = abs(p.xz);
    p.xz = (p.z>p.x) ? p.zx : p.xz;
    p.xz -= .5;
 
    vec3 q = vec3(p.z,h*p.y-.5*p.x,h*p.x+.5*p.y);
    float s = max(-q.x,0.);
    float t = clamp((q.y-.5*p.z)/(m2+.25),0.,1.);
    float a = m2*(q.x+s)*(q.x+s)+q.y*q.y;
    float b = m2*(q.x+.5*t)*(q.x+.5*t) +(q.y-m2*t)*(q.y-m2*t);
    float d2 = min(q.y,-q.x*m2-q.y*.5) > 0. ? 0. : min(a,b);
    return sqrt((d2+q.z*q.z)/m2) * sign(max(q.z,-p.y));
}

float tetrahedron(vec3 p,float h) {
     vec3 q = abs(p);
     float y = p.y;
     float d1 = q.z-max(y,0.);
     float d2 = max(q.x*.5+y*.5,0.)-min(h,h+y);
     return length(max(vec2(d1,d2),.005)) + min(max(d1,d2),0.);
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



float octahedron(vec3 p,float s) {

    p = abs(p);

    float m = p.x + p.y + p.z - s;
    vec3 q;

    if(3.0 * p.x < m) {
       q = vec3(p.x,p.y,p.z);  
    } else if(3.0 * p.y < m) {
       q = vec3(p.y,p.z,p.x); 
    } else if(3.0 * p.z < m) { 
       q = vec3(p.z,p.x,p.y);
    } else { 
       return m * 0.57735027;
    }

    float k = clamp(0.5 *(q.z-q.y+s),0.0,s);
    return length(vec3(q.x,q.y-s+k,q.z - k)); 
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

vec2 res = vec2(1.0,0.0);

vec3 q,v,l;
q = p; v = p; l = p;

float sa,b,c,s;

l.zy *= rot(time*.12);
l.xy = radmod(l.xy,12);
l.y -= 8.;

p.xz *= rot(time*.1);
p.xy = radmod(p.xy,8);
p.y -= 4.;

q.xz *= rot(time*.11);
q.xy = radmod(q.xy,24);
q.y -= 12.;

v.xz *= rot(time*.05);
v.yz *= rot(time*.01);
v.xy = radmod(v.xy,4);
v.y -= 8.;

sa = solidAngle(l,vec2(1.,.5),1.);
b = box(p,vec3(.25));
c = cylinder(q,vec3(1.));
s = sphere(v,.1);

res = opu(res,vec2(sa,5.)); 
res = opu(res,vec2(b,2.));
res = opu(res,vec2(c,12.));
res = opu(res,vec2(s,6.));

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

float calcAO(vec3 p,vec3 n) {

    float o = 0.;
    float s = 1.;

    for(int i = 0; i < 15; i++) {

        float h = .01 + .125 * float(i) / 4.; 
        float d = scene(p + h * n).x;  
        o += (h-d) * s;
        s *= .9;
        if(o > .33) break;
    
     }
     return clamp(1. - 3. * o ,0.0,1.0) * (.5+.5*n.y);   
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
    vec3 linear = vec3(.5);
    vec3 r = reflect(rd,n); 
    float amb = sqrt(clamp(.5+.5*n.x,0.,1.));
    float fre = pow(clamp(1.+dot(n,rd),0.,1.),2.);
    vec3 col = vec3(.5);

    vec3 l = normalize(vec3(10.,0.,10.));
    
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

        if(d.y == 5.) {

            p *= .25;

            col += fmCol(dd(p),vec3(f3(p),h11(45.),h11(124.)),
                   vec3(h11(235.),f3(p),h11(46.)),
                   vec3(h11(245.),h11(75.),f3(p)),
                   vec3(1.));

            col += mix(col,cell(p+f3(p*sin3(p,h11(100.)*45.
            )),12.)*col,rd.y*rd.x*col.z)*.01;
        
            ref = vec3(0.005);     

    
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
vec3 ro = vec3(0.,1.,5.);

for(int k = 0; k < AA; k++ ) {
   for(int l = 0; l < AA; l++) {
   
       vec2 o = vec2(float(k),float(l)) / float(AA) * .5;
       vec2 uv = (2.* (gl_FragCoord.xy+o) - resolution.xy)/resolution.y;

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
