#version 330 core 

out vec4 fragColor;

uniform vec2 resolution;
uniform float time;

#define SEED 120

#define AA 2
#define EPS 0.0001
#define STEPS 45
#define FOV 2.
#define VFOV 1.
#define NEAR 0.
#define FAR 32.

float dot2(vec2 v) { return dot(v,v); }
float dot2(vec3 v) { return dot(v,v); }
float ndot(vec2 a,vec2 b) { return a.x * b.x - a.y * b.y; }

//pcg,reasonable on pareto curve
float h11(float p) {
    uint st = uint(p) * 747796405u + 2891336453u; 
    uint wd = ((st >> ((st >> 28u) + 4u)) ^ st) * 277803737u;
    uint h = (wd >> 22u) ^ wd;
    return float(h) *  (1./float(uint(0xffffffff))); 
}

float h31(vec3 p) {
    p = 17.*fract(p*.46537+vec3(.11,.17,.13));
    return fract(p.x*p.y*p.z*(p.x+p.y+p.z));
}

vec3 h33(vec3 p) {
   uvec3 h = uvec3(ivec3(  p)) * 
   uvec3(1391674541U,SEED,2860486313U);
   h = (h.x ^ h.y ^ h.z) * uvec3(1391674541U,SEED,2860486313U);
   return vec3(h) * (1.0/float(0xffffffffU));

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

float n3(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f * f * (3.0 - 2.0 * f);
    float q = p.x + p.y * 157.0 + 113.0 * p.z;

    return mix(mix(mix(h11(q + 0.0),h11(q + 1.0),f.x),
           mix(h11(q + 157.0),h11(q + 158.0),f.x),f.y),
           mix(mix(h11(q + 113.0),h11(q + 114.0),f.x),
           mix(h11(q + 270.0),h11(q + 271.0),f.x),f.y),f.z);
}

float f3(vec3 p) {
    float q = 1.;

    mat3 m = mat3(vec2(.8,.6),-.6,
                  vec2(-.6,.8),.6,
                  vec2(-.8,.6),.8);

    q += .5      * n3(p); p = m*p*2.01;
    q += .25     * n3(p); p = m*p*2.04;
    q += .125    * n3(p); p = m*p*2.048;
    q += .0625   * n3(p); p = m*p*2.05;
    q += .03125  * n3(p); p = m*p*2.07; 
    q += .015625 * n3(p); p = m*p*2.09;
    q += .007825 * n3(p); p = m*p*2.1;
    q += .003925 * n3(p);

    return q / .94;
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

float easeInOut3(float t) {
    if((t *= 2.0) < 1.0) {
        return 0.5 * t * t * t;
    } else { 
        return 0.5 * ((t -= 2.0) * t * t + 2.0);
    }
}

vec3 fm(float t,vec3 a,vec3 b,vec3 c,vec3 d) {
    return a + b * cos((radians(180.)*2.0) * (c * t + d));
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

mat2 rot(float a) {
    float c = cos(a);
    float s = sin(a);
    
    return mat2(c,-s,s,c);
}

mat3 camera(vec3 ro,vec3 ta,float r) {
     
     vec3 w = normalize(ta - ro); 
     vec3 p = vec3(sin(r),cos(r),0.);           
     vec3 u = normalize(cross(w,p)); 
     vec3 v = normalize(cross(u,w));

     return mat3(u,v,w); 
}

float re(vec3 p,float d,float h) {
    vec2 w = vec2(d,abs(p.z) - h);
    return min(max(w.x,w.y),0.) + length(max(w,0.)); 
}

float spiral(vec2 p,float n,float h) {
     float ph = pow(length(p),1./n)*32.;
     p *= mat2(cos(ph),sin(ph),sin(ph),-cos(ph));
     return h-length(p) / 
     sin((atan(p.x,-p.y)
     + radians(180.)/radians(180.)/2.))*radians(180.);
}





float plane(vec3 p,vec4 n) {
    return dot(p,n.xyz) + n.w;
}

float box(vec3 p,vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0);
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

float dfn(ivec3 i,vec3 f,ivec3 c) {
    float rad = .5*h31(i+c);
    //return length(f-vec3(c))-rad;
    return ico(f-vec3(c),rad);
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
     for(int i = 0; i < 5; i++) {
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

vec3 q = p,l = p;

p.xz *= rot(.5*easeInOut3(sin(time*.5)*.25)-.125);
q.xz *= rot(time*.25);
q.xy *= rot(time*.12);

float d = dode(p,1.);

res = opu(res,vec2(plane(l,vec4(0.,0.,1.,2.)),1.));
res = opu(res,vec2(max(ico(q,1.),d),2.));

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

vec3 calcNormal(vec3 p) {
    vec2 e = vec2(EPS,0.);
    return normalize(vec3(
    scene(p + e.xyy).x - scene(p - e.xyy).x,
    scene(p + e.yxy).x - scene(p - e.yxy).x,
    scene(p + e.yyx).x - scene(p - e.yyx).x));
}

vec3 render(vec3 ro,vec3 rd) {

       vec4 d = trace(ro,rd);

       vec3 linear = vec3(0.);
       vec3 p = ro + rd * d.x;
       vec3 n = calcNormal(p);
       vec3 r = reflect(rd,n);

       vec3 l = normalize(vec3(0.,0.,1.));

       float amb = sqrt(clamp(.5+.5*n.y,0.,1.));  
       float dif = clamp(dot(n,l),0.0,1.0);
     
       vec3 h = normalize(l-rd);
       float spe = pow(clamp(dot(n,h),0.0,1.0),16.);
       spe *= dif;
       spe *= .04 + 0.9 * pow(clamp(1.+dot(h,l),0.,1.),5.);

       vec3 c = vec3(.5);

       if(d.y >= -.5) {

           if(d.y == 1.) {
        
               float n;
               vec2 e = gl_FragCoord.xy;
               
               n = dd(p+cell(p,5.));
                
               c = fm(n+e.x*.0005*e.y*.005,
                   vec3(.5),
                   vec3(.5),
                   vec3(.5,.5,1.),
                   vec3(1.,.6,.8));

               vec3 r;
               vec2 s = vec2(.35);
               vec2 q = mod(p.xy,s)-.5*s;
               float d = length(q.xy)-.12;
               r = vec3(smoothstep(.01,.03,d));
               r += mix(r,c,f3(p)); 
               c *= r;
                           
           }

           if(d.y == 2.) {
              c = vec3(.3);
              linear += .5*spe+c;

           }

           linear += dif * vec3(.05);
           linear += amb * vec3(0.1);
           linear += .05 * spe * vec3(1.);
          
           c += linear;        
       } 
        
return c;
}


void main() { 

vec3 color = vec3(0.);

vec3 ta = vec3(0.);
vec3 ro = vec3(0.,0.,5.);

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
       vec3 rd = cm * normalize(vec3(uv.xy,2.));
           
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
