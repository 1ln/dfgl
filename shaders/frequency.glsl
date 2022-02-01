#version 330 core 

out vec4 fragColor;

uniform vec2 resolution;
uniform float time;
uniform int frame;

uniform vec4 mouse;

uniform sampler2D tex;

uniform int key_x;
uniform int key_z;
uniform int up;
uniform int dn;
uniform int lf;
uniform int ri;

#define SEED 1

#define AA 1
#define EPS 0.0001
#define STEPS 255
#define FOV 2.
#define VFOV 1.
#define NEAR 0.
#define FAR 100.

float dot2(vec2 v) { return dot(v,v); }
float dot2(vec3 v) { return dot(v,v); }
float ndot(vec2 a,vec2 b) { return a.x * b.x - a.y * b.y; }

float spiral(vec2 p,float n,float h) {
     float ph = pow(length(p),1./n)*32.;
     p *= mat2(cos(ph),sin(ph),sin(ph),-cos(ph));
     return h-length(p) / 
     sin((atan(p.x,-p.y)
     + radians(180.)/radians(180.)/2.))*radians(180.);
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

float hyperbola(vec3 p) { 

vec2 l = vec2(length(p.xz) ,-p.y);
float a = 0.5;
float d = sqrt((l.x+l.y)*(l.x+l.y)- 4. *(l.x*l.y-a)) + 0.5; 
return (-l.x-l.y+d)/2.0;

}

uint rand(uint p) {
    uint st = p * 747796405u + 2891336453u; 
    uint wd = ((st >> ((st >> 28u) + 4u)) ^ st) * 277803737u;
    return (wd >> 22u) ^ wd;
}

float h31(vec3 p) {
    p = 17.*fract(p*.46537+vec3(.11,.17,.13));
    return fract(p.x*p.y*p.z*(p.x+p.y+p.z));
}

#ifdef HASH_SINE

float h11(float p) {
    return fract(sin(p)*float(43758.5453+SEED));
}
#else

float h11(float p) {
    uvec2 n = uint(int(p)) * uvec2(1391674541U,SEED);
    uint h = (n.x ^ n.y) * 1391674541U;
    return float(h) * (1./float(0xffffffffU));
}
#endif

vec3 h33(vec3 p) {
   uvec3 h = uvec3(ivec3(  p)) * 
   uvec3(1391674541U,SEED,2860486313U);
   h = (h.x ^ h.y ^ h.z) * uvec3(1391674541U,SEED,2860486313U);
   return vec3(h) * (1.0/float(0xffffffffU));

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
  
    p1 = permute(mod289(p1 + vec3(float(SEED))));

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

float sin3(vec3 p,float h) {
    return sin(p.x*h)*sin(p.y*h)*sin(p.z*h);
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

float n(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f * f * (3.0 - 2.0 * f);
    float q = p.x + p.y * 157.0 + 113.0 * p.z;

    return mix(mix(mix(h11(q + 0.0),h11(q + 1.0),f.x),
           mix(h11(q + 157.0),h11(q + 158.0),f.x),f.y),
           mix(mix(h11(q + 113.0),h11(q + 114.0),f.x),
           mix(h11(q + 270.0),h11(q + 271.0),f.x),f.y),f.z);
}

float f(vec3 p) {
    float q = 1.;

    mat3 m = mat3(vec2(.8,.6),-.6,
                  vec2(-.6,.8),.6,
                  vec2(-.8,.6),.8);

    q += .5      * n(p); p = m*p*2.01;
    q += .25     * n(p); p = m*p*2.04;
    q += .125    * n(p); p = m*p*2.048;
    q += .0625   * n(p); p = m*p*2.05;
    q += .03125  * n(p); p = m*p*2.07; 
    q += .015625 * n(p); p = m*p*2.09;
    q += .007825 * n(p); p = m*p*2.1;
    q += .003925 * n(p);

    return q / .94;
}

float dd(vec3 p) {
    vec3 q = vec3(f(p+vec3(0.,1.,2.)),
                  f(p+vec3(4.,2.,3.)),
                  f(p+vec3(2.,5.,6.)));
    vec3 r = vec3(f(p + 4. * q + vec3(4.5,2.4,5.5)),
                  f(p + 4. * q + vec3(2.25,5.,2.)),
                  f(p + 4. * q + vec3(3.5,1.5,6.)));
    return f(p + 4. * r);
}

float expStep(float x,float k) {
    return exp((x*k)-k);
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
    float a = radians(180.) * (k * x - 1.0);
    return sin(a)/a;
}

float easeIn4(float t) {
    return t * t;
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

vec3 rgbHsv(vec3 c) {
    vec3 rgb = clamp(abs(
    mod(c.x * 6. + vec3(0.,4.,2.),6.)-3.)-1.,0.,1.);

    rgb = rgb * rgb * (3. - 2. * rgb);
    return c.z * mix(vec3(1.),rgb,c.y);
}

vec3 wbar(vec2 uv,vec3 fcol,vec3 col,float y,float h) {
    return mix(fcol,col,step(abs(uv.y-y),h));
}

vec3 hbar(vec2 uv,vec3 fcol,vec3 col,float x,float w) {
    return mix(fcol,col,step(abs(uv.x-x),w));
}
 
float ls(float a,float b,float t,float n) {
     float f = mod(t,n);
     return clamp((f-a)/(b-a),0.,1.);
}

vec3 rl(vec3 p,float c,vec3 l) { 
    vec3 q = p - c * clamp( floor((p/c)+0.5) ,-l,l);
    return q; 
}

vec3 rp(vec3 p,vec3 s) {
    vec3 q = mod(p,s) - 0.5 * s;
    return q;
} 

vec2 atanp(vec2 p,float r) {
    float n = radians(360.)/r;
    float a = atan(p.x,p.y)+n*.5;
    a = floor(a/n)*n;
    return p * mat2(cos(a),-sin(a),sin(a),cos(a));
}

vec2 opu(vec2 d1,vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
} 

float smax(float d1,float d2,float k) {
    float h = clamp(0.5 - 0.5 * (d2+d1)/k,0.0,1.0);
    return mix(d2,-d1,h) + k * h * (1.0 - h);
}

float smin(float d1,float d2,float k) {

    float h = clamp(0.5 + 0.5 * (d2-d1)/k,0.0,1.0);
    return mix(d2,d1,h) - k * h * (1.0 - h);
}

float smin_exp(float d1,float d2,float k) {
    float res = exp2(-k * d1) + exp2(-k * d2);
    return -log2(res)/k;
}

float smin_pow(float d1,float d2,float k) {
     d1 = pow(d1,k);
     d2 = pow(d2,k);
     return pow((d1*d2) / (d1+d2),1./k);
}  

vec2 blend(vec2 d1,vec2 d2,float k) {
    float d = smin(d1.x,d2.x,k);
    float m = mix(d1.y,d2.y,clamp(d1.x-d,0.,1.));
    return vec2(d,m);
}

float re(vec3 p,float d,float h) {
    vec2 w = vec2(d,abs(p.z) - h);
    return min(max(w.x,w.y),0.) + length(max(w,0.)); 
} 

vec2 rv(vec3 p,float w,float f) {
    return vec2(length(p.xz) - w * f,p.y);
} 

vec3 tw(vec3 p,float k) {
    
    float s = sin(k * p.y);
    float c = cos(k * p.y);
    mat2 m = mat2(c,-s,s,c);
    return vec3(m * p.xz,p.y);
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

mat3 camera(vec3 ro,vec3 ta,float r) {
     
     vec3 w = normalize(ta - ro); 
     vec3 p = vec3(sin(r),cos(r),0.);           
     vec3 u = normalize(cross(w,p)); 
     vec3 v = normalize(cross(u,w));

     return mat3(u,v,w); 
}

float circle(vec2 p,float r) {
    return length(p) - r;
}

float circle3(vec2 p,vec2 a,vec2 b,vec2 c,float rad) {
    float d = distance(a,b);
    float d1 = distance(b,c);
    float d2 = distance(c,a);

    float r = (d-d1+d2)*rad;
    float r1 = (d+d1-d2)*rad;
    float r2 = (-d+d1+d2)*rad;

    float de = .0005;
    de = min(de,abs(distance(p,a)-r));
    de = min(de,abs(distance(p,b)-r1));
    de = min(de,abs(distance(p,c)-r2));
    return de;
}

float arc(vec2 p,vec2 sca,vec2 scb,float ra,float rb) {
    p *= mat2(sca.x,sca.y,-sca.y,sca.x);
    p.x = abs(p.x);
    float k = (scb.y*p.x>scb.x*p.y) ? dot(p,scb) : length(p);
    return sqrt(dot(p,p)+ra*ra-2.*ra*k)-rb;
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

float segment(vec2 p,vec2 a,vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa,ba)/dot(ba,ba),0.,1.);  
    return length(pa - ba * h);
}

float pentagon(vec2 p,float r) {
    const vec3 k = vec3(.809016994,.587785252,.726542528);
    p.x = abs(p.x);
    p -= 2.*min(dot(vec2(-k.x,k.y),p),0.)*vec2(-k.x,k.y);
    p -= 2.*min(dot(vec2(k.x,k.y),p),0.)*vec2(k.x,k.y);
    p -= vec2(clamp(p.x,-r*k.z,r*k.z),r);
    return length(p)*sign(p.y);
}

float sphere(vec3 p,float r) {
    return length(p) - r;
}

float plane(vec3 p,vec4 n) {
    return dot(p,n.xyz) + n.w;
}

float vertcap(vec3 p,float h,float r) {
    p.y -= clamp(p.y,0.,h);
    return length(p)-r; 
}

float capsule(vec3 p,vec3 a,vec3 b,float r) {
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0);
    return length(pa - ba * h) - r;
} 

float box(vec3 p,vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0);
}

float torus(vec3 p,vec2 t) { 
    vec2 q = vec2(length(vec2(p.x,p.z)) - t.x,p.y);
    return length(q) - t.y; 
}


float cylinder(vec3 p,float h,float r) {
    vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
    return min(max(d.x,d.y),0.) + length(max(d,0.));
}

float hex(vec3 p,vec2 h) {
 
    const vec3 k = vec3(-0.8660254,0.5,0.57735);
    p = abs(p); 
    p.xy -= 2.0 * min(dot(k.xy,p.xy),0.0) * k.xy;
 
    vec2 d = vec2(length(p.xy 
           - vec2(clamp(p.x,-k.z * h.x,k.z * h.x),h.x))
           * sign(p.y-h.x),p.z-h.y);

    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float tetrahedron(vec3 p,float h) {
     vec3 q = abs(p);
     float y = p.y;
     float d1 = q.z-max(y,0.);
     float d2 = max(q.x*.5+y*.5,0.)-min(h,h+y);
     return length(max(vec2(d1,d2),.005)) + min(max(d1,d2),0.);
}

float dode(vec3 p,float r) {
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

float dfn(ivec3 i,vec3 f,ivec3 c) {
    float rad = .5*h31(i+c);
    return length(f-vec3(c))-rad;
}

float base_df(vec3 p) {
     ivec3 i = ivec3(floor(p));
     vec3 f = fract(p);

     return min(min(min(dfn(i,f,ivec3(0,0,0)),
                        dfn(i,f,ivec3(0,0,1))),
                    min(dfn(i,f,ivec3(0,1,0)),
                        dfn(i,f,ivec3(0,1,1)))),
                min(min(dfn(i,f,ivec3(1,0,0)),
                        dfn(i,f,ivec3(1,0,1))),
                    min(dfn(i,f,ivec3(1,1,0)),
                        dfn(i,f,ivec3(1,1,1)))));
}

float base_fractal(vec3 p,float d) {
     float s = 1.;
     for(int i = 0; i < 8; i++) {
          float n = s*base_df(p);
          n = smax(n,d-.1*s,.3*s);
          d = smin(n,d,.3*s);
 
          p = mat3( 0.,1.6,1.2,
                   -1.6,.7,-.96,
                   -1.2,-.96,1.28)*p;
          s = .5*s;                      
     }
     return d;
} 


vec2 scene(vec3 p) { 

vec2 res = vec2(1.0,0.0);

#ifdef MOUSE_ROT
mat4 mx = rotAxis(vec3(1.,0.,0.),2.*radians(180.)*mouse.x);
mat4 my = rotAxis(vec3(0.,1.,0.),2.*radians(180.)*mouse.y);

p = (vec4(p,1.)*mx*my).xyz;
#endif

vec3 q = p;

p.xz *= rot(time*.1);
p.xz *= rot(.5*easeInOut3(sin(time*.5)*.25)-.125);



float ft = time*.1;
float n  = 4.;

float e1,e2,e3,e4;
e1 = easeInOut4(ls(0.,1.,ft,n));
e2 = easeIn4(ls(1.,2.,ft,n));
e3 = easeOut3(ls(2.,1.,ft,n));
e4 = easeInOut3(ls(1.,0.,ft,n));

float pl = -plane(q,vec4(1.,-1.,-1.,e1+e2+e3+e4-2.));
float d = dode(p,1.);

res = opu(res,vec2(max(pl,d),2.));

return res;

}

vec4 trace(vec3 ro,vec3 rd) { 
    float d = -1.0;
    float s = NEAR;
    float e = FAR; 

    float h = 0.;

    for(int i = 0; i < STEPS; i++) {

        vec3 p = ro + s * rd;
        vec2 dist = scene(p);
        h = float(i);   

        if(abs(dist.x) < EPS || e <  dist.x ) { break; }
        s += dist.x;
        d = dist.y;

        }

        if(e < s) { d = -1.0; }
        return vec4(s,d,h,1.);

}

float shadow(vec3 ro,vec3 rd ) {
    float res = 1.0;
    float dmax = 2.;
    float t = 0.005;
    float ph = 1e10;
    
    const float maxh = 2.;
    float e = (maxh-ro.y)/rd.y;
    if(e > 0.) dmax = min(dmax,e);    

    for(int i = 0; i < 125; i++ ) {
        
        float h = scene(ro + rd * t  ).x;

        float y = h * h / (2. * ph);
        float d = sqrt(h*h-y*y);         
        res = min(res,100. * d/max(0.,t-y));
        ph = h;
        t += h;
    
        if(res < EPS || t*rd.y+ro.y > dmax) { break; }

        }
        return clamp(res,0.0,1.0);
}

#ifdef GRADIENT 

vec3 calcNormal(vec3 p) {
    vec2 e = vec2(EPS,0.);
    return normalize(vec3(
    scene(p + e.xyy).x - scene(p - e.xyy).x,
    scene(p + e.yxy).x - scene(p - e.yxy).x,
    scene(p + e.yyx).x - scene(p - e.yyx).x));
}

#else

vec3 calcNormal(vec3 p) {
    vec2 e = vec2(1.0,-1.0) * EPS;
    return normalize(vec3(
    vec3(e.x,e.y,e.y) * scene(p + vec3(e.x,e.y,e.y)).x +
    vec3(e.y,e.x,e.y) * scene(p + vec3(e.y,e.x,e.y)).x +
    vec3(e.y,e.y,e.x) * scene(p + vec3(e.y,e.y,e.x)).x + 
    vec3(e.x,e.x,e.x) * scene(p + vec3(e.x,e.x,e.x)).x

    ));
    
}

#endif

vec3 render(vec3 ro,vec3 rd) {

       vec4 d = trace(ro,rd);

       vec3 linear = vec3(0.);
       vec3 p = ro + rd * d.x;
       vec3 n = calcNormal(p);
       vec3 r = reflect(rd,n);

       vec3 l = normalize(vec3(10.,0.,10.));
       l.xz *= rot(time*.1);
 
       float ref = smoothstep(-2.,2.,r.y);    
       float amb = sqrt(clamp(.5+.5*n.y,0.,1.));
       float fre = pow(clamp(1.+dot(n,rd),0.,1.),2.);    
       float dif = clamp(dot(n,l),0.0,1.0);
     
       vec3 h = normalize(l-rd);
       float spe = pow(clamp(dot(n,h),0.0,1.0),16.);
       spe *= dif;
       spe *= .04 + 0.9 * pow(clamp(1.+dot(h,l),0.,1.),5.);

       vec3 bg = vec3(.5);
       vec3 c = bg * max(rd.y,0.);

       if(d.y >= -.5) {

           c = vec3(.5);






           dif *= shadow(p,l);
     
           linear += dif * vec3(.5);
           linear += amb * vec3(0.01);
           linear += 5. * fre * vec3(.1);
           linear += 12. * spe * vec3(1.);
                
           c += linear;        
       }
           c = mix(c,bg,1.-exp(-.0001*d.x*d.x*d.x)); 
        
return c;
}


void main() { 

vec3 color = vec3(0.);

vec3 ta = vec3(0.);
vec3 ro = vec3(1.,2.,12.);

#if AA > 1
for(int i = 0; i < AA; i++ ) {
   for(int k = 0; k < AA; k++) {
   
       vec2 o = vec2(float(i),float(k)) / float(AA) * .5;
       vec2 uv = (2.* (gl_FragCoord.xy+o) -
       resolution.xy)/resolution.y;
#else
       vec2 uv = (2.*(gl_FragCoord.xy) -
       resolution.xy)/resolution.y;

#endif

       mat3 cm = camera(ro,ta,0.);
       vec3 rd = cm * normalize(vec3(uv.xy,5.));
           
       vec3 c = render(ro,rd);

       c = pow(c,vec3(.4545));
       color += c;
   
#if AA > 1 

   }
}   
   color /= float(AA*AA);
#endif

   fragColor = vec4(color,1.0);


}
