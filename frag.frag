#version 430 core

//Dan Olson
//2020

out vec4 FragColor;

uniform sampler2D tex;

uniform sampler2D rtex;
uniform sampler2D ntex;

uniform vec2 resolution;
uniform float time;
uniform vec2 mouse;

#define STEPS 255
#define EPS 0.00001
#define NEAR 0.
#define FAR 500

#define AA 2

#define SEED 10402592
 
mat2 dm2 = mat2(0.6,0.8,-0.6,0.8); 
mat3 dm3 = mat3(0.6,0.,0.8,0.,1.,-0.8,0.,0.6,0.);

float dot2(vec2 v) { return dot(v,v); }
float dot2(vec3 v) { return dot(v,v); }
float ndot(vec2 a,vec2 b) { return a.x * b.x - a.y * b.y; }

float h11(float p) {
    uvec2 n = uint(int(p)) * uvec2(uint(int(SEED)),2531151992.0);
    uint h = (n.x ^ n.y) * uint(int(SEED));
    return float(h) * (1./float(0xffffffffU));
}

float h21(vec2 p) {
    uvec2 n = uvec2(ivec2(p)) * uvec2(uint(int(SEED)),2531151992.0);
    uint h = (n.x ^ n.y) * uint(int(SEED));
    return float(h) * (1./float(0xffffffffU));
}

vec3 h33(vec3 p) {
   uvec3 h = uvec3(ivec3(p)) *  
   uvec3(uint(int(SEED)),2531151992.0,2860486313U);

   h = (h.x ^ h.y ^ h.z) * 
   uvec3(uint(int(SEED)),2531151992U,2860486313U);

   return vec3(h) * (1.0/float(0xffffffffU));
}

vec2 rndPntCircle(float p) {
    float a = radians(180) * h11(p);
    return vec2(cos(a),sin(a)); 
}

float fib(float n) {
    return pow(( 1. + sqrt(5.)) /2.,n) -
           pow(( 1. - sqrt(5.)) /2.,n) / sqrt(5.); 
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
    float a = radians(180) * (k * x - 1.0);
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

float n2(vec2 x) {
    vec2 p = floor(x);
    vec2 f = fract(x);

    f = f * f * (3.0 - 2.0 * f);  
    float n = p.x + p.y * 57.;  

    return mix(mix(h11(n+0.),h11(n+1.),f.x),
               mix(h11(n+57.),h11(n+58.),f.x),f.y);  
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

float f2(vec2 x,int octaves,float hurst) {
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

float dd(vec2 p,int octaves,float hurst) {
   vec2 q = vec2(f2(p+vec2(3.,0.5),octaves,hurst),
                 f2(p+vec2(1.,2.5),octaves,hurst));

   vec2 r = vec2(f2(p + 4. * q + vec2(7.5,4.35),octaves,hurst),
                 f2(p + 4. * q + vec2(5.6,2.2),octaves,hurst)); 

   return f2(p + 4. * r,octaves,hurst);
}

vec3 fmCol(float t,vec3 a,vec3 b,vec3 c,vec3 d) {
    return a + b * cos((radians(180)*2.0) * (c * t + d));
}

vec3 rgbHsv(vec3 c) {
    vec3 rgb = clamp(abs(
    mod(c.x * 6. + vec3(0.,4.,2.),6.)-3.)-1.,0.,1.);

    rgb = rgb * rgb * (3. - 2. * rgb);
    return c.z * mix(vec3(1.),rgb,c.y);
}

vec3 contrast(vec3 c) {
return smoothstep(0.,1.,c);
}

mat2 rot(float a) {

    float c = cos(a);
    float s = sin(a);
    
    return mat2(c,-s,s,c);
}

mat3 rotAxis(vec3 axis,float theta) {

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

mat3 camOrthographic(vec3 ro,vec3 ta,float r) {
     
     vec3 w = normalize(ta - ro); 
     vec3 p = vec3(sin(r),cos(r),0.);           
     vec3 u = normalize(cross(w,p)); 
     vec3 v = normalize(cross(u,w));

     return mat3(u,v,w); 
} 

mat3 camEuler(float yaw,float pitch,float roll) {

     vec3 f = -normalize(vec3(sin(yaw),sin(pitch),cos(yaw)));
     vec3 r = normalize(cross(f,vec3(0.0,1.0,0.0)));
     vec3 u = normalize(cross(r,f));

     return rotAxis(f,roll) * mat3(r,u,f);
}

vec3 repLim(vec3 p,float c,vec3 l) {
  
    vec3 q = p - c * clamp( floor((p/c)+0.5) ,-l,l);
    return q; 
}

vec2 repeat(vec2 p,float s) {
     vec2 q = mod(p,s) - .5 * s;
     return q;
}

vec3 repeat(vec3 p,vec3 s) {
   
    vec3 q = mod(p,s) - 0.5 * s;
    return q;
} 

vec3 id(vec3 p,float s) {
    return floor(p/s);
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

vec3 twist(vec3 p,float k) {
    
    float s = sin(k * p.y);
    float c = cos(k * p.y);
    mat2 m = mat2(c,-s,s,c);
    return vec3(m * p.xz,p.y);
}

float layer(float d,float h) {
    return abs(d) - h;
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
    r.xy = (p.x > 0.) ? r.xy : r.xz;
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

float nsphere(vec3 p,float r) {

    return abs(length(p)-r);
}

float ellipsoid(vec3 p,vec3 r) {

    float k0 = length(p/r); 
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}

float cone(vec3 p,vec2 c) {

    float q = length(p.xy);
    return dot(c,vec2(q,p.z));
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

float link(vec3 p,float le,float r1,float r2) {

    vec3 q = vec3(p.x,max(abs(p.y) -le,0.0),p.z);
    return length(vec2(length(q.xy)-r1,q.z)) - r2;
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
    return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0);
}

float boundingBox(vec3 p,vec3 b,float e) {
    p = abs(p)-b;
    vec3 q = abs(p+e)-e;
 
    return min(min(
        length(max(vec3(p.x,q.y,q.z),0.)) 
        + min(max(p.x,max(q.y,q.z)),0.),
        length(max(vec3(q.x,p.y,q.z),0.))+ 
        min(max(q.x,max(p.y,q.z)),0.)),
        length(max(vec3(q.x,q.y,p.z),0.))+
        min(max(q.x,max(q.y,p.z)),0.));
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

float cylinder(vec3 p,float h,float r) {
    
    float d = length(vec2(p.x,p.z)) - r;
    d = max(d, -p.y - h);
    d = max(d, p.y - h);
    return d; 
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

float gyroid(vec3 p,float s,float b,float v,float d) {

    p *= s;
    float g = abs(dot(sin(p),cos(p.zxy))-b)/s-v;
    return max(d,g*.5);

} 

float menger(vec3 p,int n,float s,float d) {

    for(int i = 0; i < n; i++) {

        vec3 a = mod(p * s,2.)-1.;
        s *= 3.;

        vec3 r = abs(1. - 3. * abs(a)); 
       
        float b0 = max(r.x,r.y);
        float b1 = max(r.y,r.z);
        float b2 = max(r.z,r.x);

        float c = (min(b0,min(b1,b2)) - 1.)/s;         
        d = max(d,c);
     }

     return d;
}

vec2 scene(vec3 p) {

    vec2 res = vec2(1.,0.);

    float d = 0.;     
    float s = 0.1;
    float t = time;  
    
    vec3 q = p;

    //p.xz *= rot2(t*s);
    //p.zy *= rot2(t*s);

    //p = repeat(p,vec3(10.));  

    d = box(p,vec3(1.));
    d = menger(p,5,1.,box(p,vec3(1.)));
    d = gyroid(p,6.,.5,.05,box(p,vec3(1.)));
    d = menger(p,4,2.,octahedron(p,1.));
    d = menger(p,4,1.,box(p,vec3(1.))); 
    d = gyroid(p,5.,.35,.015,sphere(p,1.));
    d = gyroid(p,6.,.5,.05,box(p,vec3(1.)));
    d = mix(box(p,vec3(1.)),sphere(p,1.),-2.);    
    d = max(-torus(p,vec2(1.5,.61)),box(p,vec3(1.))); 
    d = trefoil(p,vec2(1.5,.25),3.,.25,.5); 
    d = circle(rev(p,0.,pow(2.,1./3.)),1.);
    d = max(-cylinder(p,1.,0.5),box(p,vec3(1.)));

    //d = max(-plane(q*.5,vec4(1.,-1.,-1.,0.)),d);
    res = opu(res,vec2(d,2.)); 

    float pl = plane(q+vec3(0.,1.5,0.),vec4(0.,1.,0.,1.));
    res = opu(res,vec2(pl,1.));
  
  return res;

}

vec2 rayScene(vec3 ro,vec3 rd) {
    
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

vec3 fog(vec3 col,vec3 fog_col,float fog_dist,float fog_de) {
    float fog_depth = 1. - exp(-fog_dist * fog_de);
    return mix(col,fog_col,fog_depth);
}

vec3 scatter(vec3 col,vec3 tf,vec3 ts,vec3 rd,vec3 l,float de) {
    float fog_depth  = 1. - exp(-0.000001 * de);
    float light_depth = max(dot(rd,l),0.);
    vec3 fog_col = mix(tf,ts,pow(light_depth,8.));
    return mix(col,fog_col,light_depth);
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

float shadow(vec3 ro,vec3 rd) {

    float res = 1.0;
    float t = 0.005;
    float ph = 1e10;
    
    for(int i = 0; i < 45; i++ ) {
        
        float h = scene(ro + rd * t  ).x;

        float y = h * h / (2. * ph);
        float d = sqrt(h*h-y*y);         
        res = min(res,25. * d/max(0.,t-y));
        ph = h;
        t += h;
    
        if(res < EPS || t > 100.) { break; }

        }

        return clamp(res,0.0,1.0);

}

vec3 calcNormal(vec3 p) {

    vec2 e = vec2(1.,-1.)*EPS;

    return normalize(vec3(
    vec3(e.x,e.y,e.y) * scene(p + vec3(e.x,e.y,e.y)).x +
    vec3(e.y,e.x,e.y) * scene(p + vec3(e.y,e.x,e.y)).x +
    vec3(e.y,e.y,e.x) * scene(p + vec3(e.y,e.y,e.x)).x + 
    vec3(e.x,e.x,e.x) * scene(p + vec3(e.x,e.x,e.x)).x

    ));
    
}

vec3 rayCamDir(vec2 uv,vec3 camPosition,vec3 camTarget,float fPersp) {

     vec3 camForward = normalize(camTarget - camPosition);
     vec3 camRight = normalize(cross(vec3(0.0,1.0,0.0),camForward));
     vec3 camUp = normalize(cross(camForward,camRight));


     vec3 vDir = normalize(uv.x *             
     camRight + uv.y * camUp + camForward * fPersp);  

     return vDir;
}

vec3 renderPhong(vec3 ro,vec3 rd,vec3 lp,vec3 dif,vec3 spe,vec3 k,float a) {

     vec2 d = rayScene(ro,rd); 
     vec3 p = ro + rd * d.x; 
     vec3 n = calcNormal(p);

     vec3 l = normalize(lp - p);
     vec3 v = normalize(rd - p);

     vec3 r = normalize(reflect(-l,n));
     vec3 h = normalize(l + v);  

     float nl = clamp(dot(l,n),0.0,1.0 );

     float rv = dot(r,h);

     if(nl < 0.0) { return vec3(0.0); }
     if(rv < 0.0) { return k * (dif * nl); } 
     
     return k * (dif * nl + spe * pow(rv,a));

}

vec3 renderNormals(vec3 ro,vec3 rd) {

   vec2 d = rayScene(ro,rd);
   vec3 p = ro + rd * d.x;
   vec3 n = calcNormal(p);   
   
   return n;
}

vec3 renderScene(vec3 ro,vec3 rd) {
 
vec2 d = rayScene(ro, rd);

vec3 col = vec3(1.) * max(0.,rd.y);

if(d.y >= 0.) {

vec3 p = ro + rd * d.x;
vec3 n = calcNormal(p);
vec3 l = normalize(vec3(10.));
vec3 h = normalize(l - rd);
vec3 r = reflect(rd,n);

col = 0.2 + 0.2 * sin(2.*d.y + vec3(0.,1.,2.));

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

col = mix(col,vec3(1.),1.-exp(-0.00001 * d.x*d.x*d.x)); 

}

return col;
}

void main() {

vec2 uv = 2.* resolution.xy/resolution.y;  
vec2 m = mouse / resolution.xy; 

vec3 color = vec3(0.);

vec3 ro = vec3(0.0,0.0,5.0);
vec3 ta = vec3(0.0);
  
ro.zx *= rot(-m.x * radians(180));
ro.yz *= rot(-m.y * (radians(180)*2.));

for(int k = 0; k < AA; ++k) {
    for(int l = 0; l < AA; ++l) {

    vec2 o = vec2(float(l),float(k)) / float(AA) - .5;

    vec2 uv = (2. * (gl_FragCoord.xy + o) -
              resolution.xy) / resolution.y; 

    mat3 cm = camOrthographic(ro,ta,0.);
    vec3 rd = cm * normalize(vec3(uv.xy,2.));

    vec3 render = renderScene(ro,rd);    

    color += render;
    }

color /= float(AA*AA);
color = pow(color,vec3(.4545));

vec3 rt_col = texture(tex,gl_FragCoord.xy / resolution.xy).xyz; 
vec3 rm_col = (rt_col + color);
FragColor = vec4(rm_col,1.0);
}

}
