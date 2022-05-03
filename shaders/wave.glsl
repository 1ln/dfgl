#version 330 core 

out vec4 fragColor;

uniform vec2 resolution;
uniform float time;
uniform int frame;

uniform vec2 mouse;

uniform sampler2D tex;

uniform int key_x;
uniform int key_z;
uniform int up;
uniform int dn;
uniform int lf;
uniform int ri;

//Wave
//2022

#define FC gl_FragCoord.xy
#define RE resolution
#define T time
#define S smoothstep
#define L length

#define SEED 1

#define AA 1
#define EPS 0.00001
#define STEPS 255
#define FOV 2.
#define VFOV 1.
#define NEAR 0.
#define FAR 225.

//zero for no gamma
#define GAMMA .4545

#define PI radians(180.)
#define TAU radians(180.)*2.



float dot2(vec2 v) { return dot(v,v); }
float dot2(vec3 v) { return dot(v,v); }
float ndot(vec2 a,vec2 b) { return a.x * b.x - a.y * b.y; }


#ifdef HASH_INT

float h(float p) {
    uint st = uint(p) * 747796405u + 2891336453u + uint(SEED);
    uint wd = ((st >> ((st >> 28u) + 4u)) ^ st) * 277803737u;
    uint h = (wd >> 22u) ^ wd;
    return float(h) * (1./float(uint(0xffffffff)));
}

#else

float h(float p) {
    return fract(sin(p)*float(43758.5453+SEED));
}

#endif

float cell(vec3 x,float n,int a) {
    x *= n;
    vec3 p = floor(x);
    vec3 f = fract(x);
 
    float md = 1.0;
    
    for(int i = -1; i <= 1; i++) {
        for(int j = -1; j <= 1; j++) {
            for(int k = -1; k <= 1; k++) { 

                vec3 b = vec3(float(k),float(j),float(i));
                vec3 r = vec3(h(p.x+b.x+h(p.y+b.y+h(p.z+b.z))));
                
                vec3 diff = (b + r - f);

                float d = length(diff);
    
                if(a == 0) {
                md = min(md,d);
                }   

                if(a == 1) {
                md = min(md,abs(diff.x)+ 
                            abs(diff.y)+
                            abs(diff.z));
                }          

                if(a == 2) {                
                md = min(md,max(abs(diff.x),                 
                            max(abs(diff.y),
                            abs(diff.z))));                       
                }

            }
        }
    }
 
    return md;

}

float n3(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f * f * (3.0 - 2.0 * f);
    float q = p.x + p.y * 157.0 + 113.0 * p.z;

    return mix(mix(mix(h(q + 0.0),h(q + 1.0),f.x),
           mix(h(q + 157.0),h(q + 158.0),f.x),f.y),
           mix(mix(h(q + 113.0),h(q + 114.0),f.x),
           mix(h(q + 270.0),h(q + 271.0),f.x),f.y),f.z);
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

float expstep(float x,float k) {
    return exp((x*k)-k);
}


vec3 fm(float t,vec3 a,vec3 b,vec3 c,vec3 d) {
    return a + b * cos(TAU * (c * t + d));
}

vec3 bayer(vec2 p,vec3 c,int n) {
    float f = 0.;
    for(int i = 0; i < n; i++) {
        vec2 s;
        if(i == 0) {
            s = vec2(2.);
        } else if(i == 1) {
            s = vec2(4.);
        } else if(i == 2) {
            s = vec2(8.);
        };

        vec2 t = mod(p,s)/s;
        int d = int(dot(floor(t*2.),vec2(2.,1.)));
        float b = 0.;

        if(d == 0) {
            b = 0.; 
        } else if(d == 1) {
            b = 2.;
        } else if(d == 2) {
            b = 3.;
        } else {  b = 1.; }

        if(i == 0) {
            f += b * 16.;
        } else if(i == 1) {
            f += b * 4.;
        } else if(i == 2) {
            f += b * 1.;
        }
    }

float h = f / 64.;
return vec3(
           step(h,c.r),
           step(h,c.b),
           step(h,c.g));
}

vec2 opu(vec2 d1,vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
} 

float re(vec3 p,float d,float h) {
    vec2 w = vec2(d,abs(p.z) - h);
    return min(max(w.x,w.y),0.) + length(max(w,0.)); 
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

mat3 camera(vec3 ro,vec3 ta,float r) {
     
    vec3 w = normalize(ta - ro); 
    vec3 p = vec3(sin(r),cos(r),0.);           
    vec3 u = normalize(cross(w,p)); 
    vec3 v = normalize(cross(u,w));

    return mat3(u,v,w); 
}

mat3 ra3(vec3 axis,float theta) {

axis = normalize(axis);

    float c = cos(theta);
    float s = sin(theta);

    float oc = 1.0 - c;

    return mat3(
 
        oc * axis.x * axis.x + c, 
        oc * axis.x * axis.y - axis.z * s,
        oc * axis.z * axis.x + axis.y * s, 
    
        oc * axis.x * axis.y + axis.z * s,
        oc * axis.y * axis.y + c, 
        oc * axis.y * axis.z - axis.x * s,

        oc * axis.z * axis.x - axis.y * s,
        oc * axis.y * axis.z + axis.x * s, 
        oc * axis.z * axis.z + c);

}

mat3 camEuler(float yaw,float pitch,float roll) {

    vec3 f = -normalize(vec3(sin(yaw),sin(pitch),cos(yaw)));
    vec3 r = normalize(cross(f,vec3(0.0,1.0,0.0)));
    vec3 u = normalize(cross(r,f));

    return ra3(f,roll) * mat3(r,u,f);
}

float sphere(vec3 p,float r) {
    return length(p) - r;
}


float plane(vec3 p,vec4 n) {
    return dot(p,n.xyz) + n.w;
}

float capsule(vec3 p,vec3 a,vec3 b,float r) {
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0);
    return length(pa - ba * h) - r;
} 

float prism(vec3 p,vec2 h) {
    vec3 q = abs(p);
    return max(q.z - h.y,  
    max(q.x * 0.866025 + p.y * 0.5,-p.y) - h.x * 0.5); 
}

float box(vec3 p,vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d,0.0))
    + min(max(d.x,max(d.y,d.z)),0.0);
}





#define R res = opu(res,vec2(
vec2 scene(vec3 p) { 

vec2 res = vec2(1.0,0.0);
vec3 q = p;

float scl = .105;

p.y += sin(p.z+n3(p)+22.25)*.5;

R max(0.,box(p/scl,vec3(1.,.25,1e10))*scl),1.));
R box(q-vec3(0.,-1.,0.),vec3(.5)),2.));



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

vec3 fog(vec3 c,vec3 b,float f,float d) {
    return mix(c,b,1.-exp(-f*d*d*d));
}
    
vec3 scatter(vec3 c,vec3 b,float f,float d,vec3 rd,vec3 l) {
    return mix(c,mix(b,c,pow(max(dot(rd,l),0.),8.)),
    1.-exp(-f*d*d*d));      
}

float ambient(vec3 n) {
    return sqrt(clamp(.5+.5*n.y,0.,1.));
}

float fresnel(vec3 n,vec3 rd) {
    return pow(clamp(1.+dot(n,rd),0.,1.),2.);    
}

float diffuse(vec3 n,vec3 l) {
    return clamp(dot(n,l),0.0,1.0);
} 

float specular(vec3 n,vec3 rd,vec3 l) {
    vec3 h = normalize(l-rd);
    float spe = pow(clamp(dot(n,h),0.0,1.0),16.);
    spe *= pow(clamp(1.+dot(h,l),0.,1.),5.);
    return spe;
}

vec3 render(inout vec3 ro,inout vec3 rd,inout vec3 ref,vec3 l) {

    vec3 c = vec3(0.);
    vec3 bg = vec3(0.5);   
    
    vec4 d = trace(ro,rd);
      
       if(d.x < FAR) {
        
           vec3 p = ro + rd * d.x;
           vec3 n = calcNormal(p);
           vec3 r = reflect(rd,n);
      
           float amb = ambient(n);
           float dif = diffuse(n,l);
           float spe = specular(n,rd,l);
           float fre = fresnel(n,rd);   
  
           float sh = shadow(p,l);
         
           c += dif * vec3(.5) * sh;
           c += amb * vec3(0.01);
           c += .1 * fre * vec3(.1);
           c += .5 * spe * vec3(1.);                
    
           if(d.y == 2.) { 
               c += vec3(.1);
               ref = vec3(.1);
           }

           if(d.y == 1.) {          
               c += vec3(1.,0.,0.); 
               ref = vec3(.0);
           }                  
   
          ro = p + n * EPS;
          rd = r; 

       }

c = scatter(c,bg,.01,d.x,rd,l);       
return c;

}

void main() { 

vec3 ro = vec3(2.,4.,2.);
vec3 ta = vec3(0.); 

vec3 c = vec3(0.);
vec3 fc = vec3(0.); 

vec3 r = vec3(.5);
vec3 ref = vec3(0.);

vec3 l = normalize(vec3(10.,2.,2.));

for(int i = 0; i < AA; i++ ) {
   for(int k = 0; k < AA; k++) {
   
       vec2 o = vec2(float(i),float(k)) / float(AA) * .5;
       vec2 uv = (2.* (FC+o) -
       RE.xy)/RE.y;

       mat3 cm = camera(ro,ta,0.);
       vec3 rd = cm * normalize(vec3(uv.xy,2.));

       for(int i = 0; i < 3; i++) {      
       c += render(ro,rd,ref,l)*r;
       r *= ref;    
       }

       c = pow(c,vec3(GAMMA));
       fc += c;
   }
}   
  
   fc /= float(AA*AA);

   fragColor = vec4(fc,1.0);


}
