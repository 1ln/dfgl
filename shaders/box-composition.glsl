#version 330 core

// dolson,2019

out vec4 FragColor;

uniform vec2 resolution;
uniform float time;
uniform int seed;

const int steps = 250;
float eps = 0.00001;
float dmin = 0.;
float dmax = 750.;
const int aa = 2;
 
const int shsteps = 45; 
float shblur = 125.;
float shmax = 10.;

float dot2(vec2 v) { return dot(v,v); }
float dot2(vec3 v) { return dot(v,v); }
float ndot(vec2 a,vec2 b) { return a.x * b.x - a.y * b.y; }

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

vec3 h33(vec3 p) {
   uvec3 h = uvec3(ivec3(  p)) *  
   uvec3(uint(int(seed)),2531151992.0,2860486313U);
   h = (h.x ^ h.y ^ h.z) * 
   uvec3(uint(int(seed)),2531151992U,2860486313U);
   return vec3(h) * (1.0/float(0xffffffffU));
}

float cell(vec3 x,float iterations,int type) {
    x *= iterations;
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

                    if(type == 0) { 
                        min_dist = min(min_dist,d);
                    }
 
                    if(type == 1) {
                        min_dist = min(min_dist,
                        abs(diff.x)+abs(diff.y)+abs(diff.z));
                    }

                    if(type == 2) {
                        min_dist = min(min_dist,
                        max(abs(diff.x),max(abs(diff.y),
                        abs(diff.z))));
                    }

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

    return mix(mix(mix(h11(n + 0.0), 
                       h11(n + 1.0),f.x),
                   mix(h11(n + 157.0),
                       h11(n + 158.0),f.x),f.y),
               mix(mix(h11(n + 113.0), 
                       h11(n + 114.0),f.x),
                   mix(h11(n + 270.0), 
                       h11(n + 271.0),f.x),f.y),f.z);
}

float f3(vec3 x,int octaves,float hurst) {
    float s = 0.;
    float h = exp2(-hurst);
    float f = 1.;
    float a = .5;

    for(int i = 0; i < octaves; i++) {

        s += a * n3(f * x);  
        f *= 2.;
        a *= h;
    }
    return s;
}

float ei(float x,float k) {

    float h = k * x;
    return h * exp(1.0 - h);
}

float es(float x,float k,float n) {
    return exp(-k * pow(x,n));
}

mat2 rot(float a) {

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

vec3 rl(vec3 p,float c,vec3 l) {
    vec3 q = p - c * clamp( floor((p/c)+0.5) ,-l,l);
    return q; 

}

vec2 opu(vec2 d1,vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
} 

float opu(float d1,float d2) {
    return min(d1,d2);
}

float opi(float d1,float d2) {
    return max(d1,d2);
}

float opd(float d1,float d2) {
    return max(-d1,d2);
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

float extr(vec3 p,float d,float h) {
    vec2 w = vec2(d,abs(p.z) - h);
    return min(max(w.x,w.y),0.) + length(max(w,0.)); 
} 
 
float roundRect(vec2 p,vec2 b,vec4 r) {
    r.xy = (p.x > 0.) ? r.xy : r.zw;
    r.x  = (p.y > 0.) ? r.x  : r.y;
    vec2 q = abs(p) - b + r.x;
    return min(max(q.x,q.y),0.) + length(max(q,0.)) - r.x;
}

float arch(vec2 p,vec2 c,float r,vec2 w) {
    p.x = abs(p.x);
    float l = length(p);
    p = mat2(-c.x,c.y,c.y,c.x)*p;
    p = vec2((p.y>0.)?p.x:l*sign(-c.x),
             (p.x>0.)?p.y:l);
    p = vec2(p.x,abs(p.y-r))-w;
    return length(max(p,0.)) + min(0.,max(p.x,p.y));
}
  
float box(vec3 p,vec3 b) {

    vec3 d = abs(p) - b;
    return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0);
}

float boxFrame(vec3 p,vec3 b,float e) {
    
    p = abs(p) - b;
    vec3 q = abs(p+e)-e;
    
    return min(min(
        length(max(vec3(p.x,q.y,q.z),0.)) +
           min(max(p.x,max(q.y,q.z)),0.),
        length(max(vec3(q.x,p.y,q.z),0.)) +
           min(max(p.y,max(q.x,q.z)),0.)),
        length(max(vec3(q.x,q.y,p.z),0.)) +
           min(max(p.z,max(q.x,q.y)),0.));
}

vec2 scene(vec3 p) {

    vec2 res = vec2(1.,0.);

    res = opu(res,vec2(
        box(p+vec3(0.,.05,0.),vec3(.05)),25.));   

    res = opu(res,vec2(
      smou(length(p-vec3(1.,.25,0.))-.25,
           box(p-vec3(1.,.1,0.),vec3(.25,.1,.25))
           ,.1),125.));

    res = opu(res,vec2(
        extr(p.yzx,arch(-p.yz+vec2(-.1,.75) 
        ,vec2(0.,1.),.189,vec2(1e10,.005)),.075),5.));

    vec3 l = p;
    l.x -= 1.;
    res = opu(res,vec2(
    extr(l.yzx,arch(-l.yz+vec2(-.189,.61)
    ,vec2(0.,1.),.11,vec2(1e10,.005)),.35),12.));

    res = opu(res,vec2(
        boxFrame(p,vec3(.125),.0125)
        ,75.));

    res = opu(res,vec2(
        max(p.y-.5,
        max(-extr(p,roundRect(p.xy-vec2(0.,.6),
        vec2(1.71,.75),vec4(.1)),1e10),   

        max(-extr(p,roundRect(p.xy+vec2(0.,.25),
        vec2(.25,.05),vec4(.05)),1e10), 

        max(-extr(p,roundRect(p.xy+vec2(-1.,.25),
        vec2(.5,.05),vec4(.05)),1e10),

        p.z-.61
        )))),2.));

    float scl = .05;
    vec3 q = p+vec3(1.,-.1,0.);
    q = rl(q/scl,1.5,vec3(5.,0.,0.))*scl;        
    res = opu(res,vec2(
        max(p.z-.5,box(q/scl,vec3(.25,.25,1e10))*scl),95.));

    res = opu(res,vec2(
    
        max(-box(p,vec3(.33)),
        max(-box(p+vec3(1.,-.5,0.),vec3(1.,.5,1e10)),
        extr(p,roundRect(p.xy,vec2(1.61,.05),vec4(.025)),.5)
        )),1.));

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

float calcAO(vec3 p,vec3 n) {

    float o = 0.;
    float s = 1.;

    for(int i = 0; i < 3; i++) {
 
        float h = .01 + .125 * float(i) / 4.; 
        float d = scene(p + h * n).x;  
        o += (h-d) * s;
        s *= .9;
        if(o > .33) break;
    
     }
     return clamp(1. - 3. * o ,0.0,1.0) * (.5+.5*n.y);   
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

vec3 col = vec3(1.) * max(0.,rd.y);

if(d.y >= 0.) {

    vec3 p = ro + rd * d.x;
    vec3 n = calcNormal(p);

    vec3 l = normalize(vec3(2.,5.,2.));

    vec3 h = normalize(l - rd);
    vec3 r = reflect(rd,n);

    col = 1.+.25*sin(d.y+vec3(.5));

    float nl = n3(p);

    if(d.y == 5.) {

        nl += mix(f3(p,8,sin(col.y)),
        0,step(h11(23.),h11(37.)));

        nl += mix(f3(p+f3(p,6,h11(111.)),4,h11(43.)),
        0,step(h11(161.),h11(100.)));

        nl += mix(cell(p+f3(p,6,.5),12.,int(floor(h11(124.)*2.))),
        0,step(h11(95.),h11(235.)));
 
        col = vec3(nl);

    }

    if(d.y == 12.) {
        
        nl += mix(f3(p,8,ei(nl,.5)),
        0,step(h11(231.),h11(105.)));     
    
        nl += mix(f3(p+es(n3(p),h11(116.),.5),8,h11(25.)),
        0,step(h11(112.),h11(44.)));

        col += vec3(nl); 

    }

    float amb = clamp(0.5 + 0.5 * n.y,0.,1.);
    float dif = clamp(dot(n,l),0.0,1.0);
    float spe = pow(clamp(dot(n,h),0.0,1.0),16.)
        * dif * (.04 + 0.9 * pow(clamp(1. + dot(h,rd),0.,1.),5.));
    float fre = pow(clamp(1. + dot(n,rd),0.0,1.0),2.0);
    float ref = smoothstep(-.2,.2,r.y);
    float ao = calcAO(p,n);

    vec3 linear = vec3(.5);

    dif *= shadow(p,l);
    ref *= shadow(p,r);

    linear += dif * vec3(1.25,0.5,0.11)*ao;
    linear += amb * vec3(0.005,0.05,0.05);
    linear += ref * vec3(0.05,0.001,0.005);
    linear += fre * vec3(0.25,0.5,0.35);

    col = col * linear;
    col += spe * vec3(0.01,0.097,0.001); 

    col = mix(col,vec3(1.),1.-exp(-0.00001 * d.x*d.x*d.x)); 

}

return col;
}

void main() {
 
vec3 color = vec3(0.);

vec3 ro = vec3(5.);
vec3 ta = vec3(0.);

for(int k = 0; k < aa; ++k) {
    for(int l = 0; l < aa; ++l) {

    vec2 o = vec2(float(l),float(k)) / float(aa) - .5;

    vec2 uv = (2. * (gl_FragCoord.xy + o) -
              resolution.xy) / resolution.y; 

    mat3 cm = camOrthographic(ro,ta,0.);
    vec3 rd = cm * normalize(vec3(uv.xy,5.));   

    vec3 col = renderScene(ro,rd);    

    col = pow(col,vec3(.4545)); 
    color += col;
    }
}

color /= float(aa*aa);
FragColor = vec4(color,1.0);

}
