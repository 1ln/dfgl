#version 330     
out vec4 fragColor;
uniform vec2 resolution;
uniform float time;

//108
//2021
//do

vec2 R = resolution;
float t = time;

#define AA 1
#define STEPS 128
#define EPS 0.0001
#define NEAR 0.
#define FAR 200.

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

float spiral(vec2 p,float s) {
    float d = length(p);
    float a = atan(p.x,p.y);
    float l = log(d)/.618 +a;
    return sin(l*s);
}

float sin2(vec2 p,float h) {
    return sin(p.x*h) * sin(p.y*h);
}

float sin3(vec3 p,float h) {
    
    return sin(p.x*h) * sin(p.y*h) * sin(p.z*h);
}

float easeIn4(float t) {
    return t * t;
}

float easeOut4(float t) {
    return -1.0 * t * (t - 2.0);
}

float easeIn3(float t) {
    return t * t * t;
}

float easeOut3(float t) {
    return (t = t - 1.0) * t * t + 1.0;
}

vec3 fmCol(float t,vec3 a,vec3 b,vec3 c,vec3 d) {
    return a + b * cos((radians(180.)*2.0) * (c * t + d));
}

vec2 opu(vec2 d1,vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
} 

float smod(float d1,float d2,float k) {
    float h = clamp(0.5 - 0.5 * (d2+d1)/k,0.0,1.0);
    return mix(d2,-d1,h) + k * h * (1.0 - h);
}

float extr(vec3 p,float d,float h) {
    vec2 w = vec2(d,abs(p.z) - h);
    return min(max(w.x,w.y),0.) + length(max(w,0.)); 
} 

vec2 rmod(vec2 p,float r) {
    float n = radians(360.)/r;
    float a = atan(p.x,p.y)+n*.5;
    a = floor(a/n)*n;
    return p * mat2(cos(a),-sin(a),sin(a),cos(a));
}

float cell(vec2 x,float n) {
    x *= n;
    vec2 p = floor(x);
    vec2 f = fract(x);
    float min_dist = 1.;
    
    for(int i = -1; i <= 1; i++) {
        for(int j = -1; j <= 1; j++) {

        vec2 b = vec2(float(j),float(i));
        vec2 r = h22(p+b);
        vec2 diff = (b+r-f);
        float d = length(diff);
        min_dist = min(min_dist,d);
        }
    }
    return min_dist;
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

float f(vec3 p) {
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

float dd(vec3 p) {
    vec3 q = vec3(f(p+vec3(0.,1.,2.)),
                  f(p+vec3(4.,2.,3.)),
                  f(p+vec3(2.,5.,6.)));
    vec3 r = vec3(f(p + 4. * q + vec3(4.5,2.4,5.5)),
                  f(p + 4. * q + vec3(2.25,5.,2.)),
                  f(p + 4. * q + vec3(3.5,1.5,6.)));
    return f(p + 4. * r);
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

float petal(vec2 p,float r1,float r2,float h) {
    p.x = abs(p.x);
    float b = (r1-r2)/h;
    float a = sqrt(1.-b*b);
    float k = dot(p,vec2(-b,a));
    if(k < 0.) return length(p)-r1;
    if(k > a*h) return length(p-vec2(0.,h))-r2;
    return dot(p,vec2(a,b))-r1;
}

float arc(vec2 p,vec2 sca,vec2 scb,float ra,float rb) {
    p *= mat2(sca.x,sca.y,-sca.y,sca.x);
    p.x = abs(p.x);
    float k = (scb.y*p.x>scb.x*p.y) ? dot(p,scb) : length(p);
    return sqrt(dot(p,p)+ra*ra-2.*ra*k)-rb;
}

float box(vec3 p,vec3 b) {

    vec3 d = abs(p) - b;
    return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0);
}

float dode(vec3 p,vec3 a,vec3 b) {
   vec4 v = vec4(0.,1.,-1.,0.5 + sqrt(1.25));
   v /= length(v.zw);

   float d;
   d = abs(dot(p,v.xyw))-a.x;
   d = max(d,abs(dot(p,v.ywx))-a.y);
   d = max(d,abs(dot(p,v.wxy))-a.z);
   d = max(d,abs(dot(p,v.xzw))-b.x);
   d = max(d,abs(dot(p,v.zwx))-b.y);
   d = max(d,abs(dot(p,v.wxz))-b.z);
   return d;
}
 
float ico(vec3 p,float r) {
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
vec3 q,y,l,k,n;

vec3 e = vec3(1./phi,1.,.67);

k = p;
k.xz *= rot(t*.61);

float d = dode(k,vec3(e.z),vec3(e.z));
d = max(d,-dode(k,e.yxx,vec3(e.x)));
d = max(d,-dode(k,e.xyx,vec3(e.x)));
d = max(d,-dode(k,e.xxy,vec3(e.x)));
d = max(d,-dode(k,vec3(e.x),e.yxx));
d = max(d,-dode(k,vec3(e.x),e.xyx));
d = max(d,-dode(k,vec3(e.x),e.xxy));

res = opu(res,vec2(d,5.));

l = p;
l.yz *= rot(t*.5);
res = opu(res,vec2(ico(l,1./phi),20.));

float an1 = pi * (.5+.5*cos(.5));
float an2 = pi * (.5+.5*cos(1.));
float rb  = .1 * (.5+.5*cos(5.));

res = opu(res,vec2(
    extr(p.xzy,arc(p.xz,vec2(sin(an1),cos(an1)),
                 vec2(sin(an2),cos(an2)),1.25,rb),
                 .1),2.));

an1 = pi * (-.5+.5*cos(.502));
res = opu(res,vec2(
    extr(p.xzy,arc(p.xz,vec2(sin(an1),cos(an1)),
    vec2(sin(an2),cos(an2)),1.5,rb),
    .16),0.));

y = p.xzy;
y.xy = rmod(y.xy,8.);
y.y -= 6.;

vec2 c = abs(vec2(length(p.xz),p.y)) - vec2(4.,2.);
float cy = min(max(c.x,c.y),0.) + length(max(c,0.));

res = opu(res,vec2(
    max(-cy,box(y,vec3(1.,1e10,1.))),8.));

n = p.xzy;
n.xy = rmod(n.xy,108.);
n.y -= 3.3;

res = opu(res,vec2(
   length(n-vec3(0.,0.,.08))-.08,108.));

q = p.xzy;
q.xy = rmod(q.xy,24.);
q.y -= 2.;

res = opu(res,vec2(    
    max(-extr(q,petal(q.xy,.24,.12,.5),.1),
    max(-extr(p.xzy-vec3(0.,0.,.05),abs(length(p.xz)-2.5)-.5,.05),
    extr(p.xzy,abs(length(p.xz)-3.)-2.,.05) 
    ))
    ,1.));




return res;


}

vec2 trace(vec3 ro,vec3 rd) {
    
    float d = -1.0;
    float s = 0.;
    float e = 225.;  

    for(int i = 0; i < 124; i++) {

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

vec3 render(vec3 ro,vec3 rd) {

    vec2 d = trace(ro,rd); 

    vec3 p = ro + rd * d.x;

    vec3 n = calcNormal(p);
    vec3 r = reflect(rd,n);    

    vec3 linear = vec3(0.);

    float amb = sqrt(clamp(.5+.5*n.y,0.,1.));
    float fre = pow(clamp(1.+dot(n,rd),0.,1.),2.);
    float re = smoothstep(-2.,2.,r.y);

    vec3 col = vec3(.5);
    vec3 bg_col = vec3(1.);
    col = bg_col * max(1.,rd.y);

    vec3 l = normalize(vec3(10.));
    vec3 h = normalize(l - rd);  

    float dif = clamp(dot(n,l),0.0,1.0);
    float spe = pow(clamp(dot(n,h),0.0,1.0),16.)
    * dif * (.04 + 0.9 * pow(clamp(1. + dot(h,rd),0.,1.),5.));

    if(d.y >= 0.) {

        dif *= shadow(p,l);
        re  *= shadow(p,r);

        linear += dif * vec3(.5);

        linear += amb * vec3(0.0001);
        linear += fre * vec3(.005,.002,.001);
        linear += spe * vec3(0.001,0.001,.005)*re;

        if(d.y == 108.) {
        col = vec3(1./phi);
        }

        if(d.y == 2.) {
        col = vec3(3.);
        }

        if(d.y == 0.) {
        col = vec3(0.,0.001,0.);
        }
        
        if(d.y == 8.) {
            col = vec3(.5,1.,.2);
        }

        if(d.y == 20.) {

            p *= .25;

            col += fmCol(dd(p),vec3(f3(p),h11(45.),h11(124.)),
                   vec3(h11(235.),f3(p),h11(46.)),
                   vec3(h11(245.),h11(75.),f3(p)),
                   vec3(1.));

            col += mix(col,cell(p+f3(p*sin3(p,h11(100.)*45.
            )),12.)*col,rd.y*rd.x*col.z)*.01;     
    
        }    

        if(d.y == 5.) {
            col = vec3(.1,.5,.25);
        }
        
        col = col * linear;
        col = mix(col,bg_col,1.-exp(-.00001*d.x*d.x));         

   }

return col;
}

void main() { 

vec3 color = vec3(0.);

vec3 ta = vec3(0.1);
vec3 ro = vec3(1.,3.,2.);
ro.xz *= rot(t*.005);





vec2 uv = (2.* (gl_FragCoord.xy)
- R.xy)/R.y;

vec3 rd = raydir(uv,ro,ta,1.);
vec3 ref = vec3(0.);
vec3 col = render(ro,rd);       
col = pow(col,vec3(.4545));
color += col;
fragColor = vec4(color,1.0);

}
