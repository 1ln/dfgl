#version 330 core     

//Dan Olson
//2020

out vec4 FragColor;

uniform sampler2D tex;

uniform vec2 resolution;
uniform float time;

uniform vec2 mouse;
uniform float mouse_scroll;
uniform int mouse_pressed;

uniform int seed;

const int steps = 250;
float eps = 0.00001;
float dmin = 0.;
float dmax = 750.;
const int aa = 2;
 
const int shsteps = 45; 
float shblur = 10.0;
float shmax = 100.; 

const int aosteps = 5;

const int octaves = 5;
float hurst = 0.5;

const float PI   =  radians(180.0); 
const float PI2  =  PI * 2.;
const float PHI  =  (1.0 + sqrt(5.0)) / 2.0;

mat2 dm2 = mat2(0.6,0.8,-0.6,0.8); 
mat3 dm3 = mat3(0.6,0.,0.8,0.,1.,-0.8,0.,0.6,0.);

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

float n2(vec2 x) { 

    vec2 p = floor(x);
    vec2 f = fract(x);

    f = f * f * (3.0 - 2.0 * f);  
    float n = p.x + p.y * 57.;  

    return mix(mix(h11(n+0.),h11(n+1.),f.x),
               mix(h11(n+57.),h11(n+58.),f.x),f.y);  
}

float f2(vec2 x) {

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

float dd(vec2 p) {

   vec2 q = vec2(f2(p+vec2(3.,0.5)),
                 f2(p+vec2(1.,2.5)));

   vec2 r = vec2(f2(p + 4. * q + vec2(7.5,4.35)),
                 f2(p + 4. * q + vec2(5.6,2.2))); 

   return f2(p + 4. * r);
}

vec3 fmCol(float t,vec3 a,vec3 b,vec3 c,vec3 d) {
    return a + b * cos( (PI*2.0) * (c * t + d));
}

mat2 rot2(float a) {

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

vec3 repLimit(vec3 p,float c,vec3 l) {
  
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

vec3 twist(vec3 p,float k) {
    
    float s = sin(k * p.y);
    float c = cos(k * p.y);
    mat2 m = mat2(c,-s,s,c);
    return vec3(m * p.xz,p.y);
}

float layer(float d,float h) {
    return abs(d) - h;
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
    return max(q.z - h.y,max(q.x * 0.866025 + p.y * 0.5,-p.y) - h.x * 0.5); 
}

float box(vec3 p,vec3 b) {

    vec3 d = abs(p) - b;
    return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0);
}

float roundbox(vec3 p,vec3 b,float r) {

    vec3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
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

vec2 scene(vec3 p) {

    vec2 res = vec2(1.,0.);

    float d = 0.;     
    float t = time;  
    
    vec3 q = p;
    vec3 pl = p; 

    p.xz *= rot2(t);
    p.zy *= rot2(t);

    float scale = 0.05;
    q = repLimit(p/scale,4.,vec3(5.))*scale;  
    d = box(q/scale,vec3(1.))*scale;
    res = opu(res,vec2(d,0.25)); 

    res = opu(res,
    vec2(plane(pl+vec3(0.,5.,1.),
    vec4(1.)),0.45));

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

    for(int i = 0; i < aosteps; i++) {
 
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

vec3 renderScene(vec3 ro,vec3 rd,vec3 lp,vec3 c) {
 
vec2 d = rayScene(ro, rd);

vec3 col = c * max(0.,rd.y);

if(d.y >= 0.) {

vec3 p = ro + rd * d.x;
vec3 n = calcNormal(p);
vec3 l = normalize(lp);
vec3 h = normalize(l - rd);
vec3 r = reflect(rd,n);

col = 0.2 + 0.2 * sin(2.*d.y + vec3(1.,3.,2.));

if(d.y == 0.45) {
    col += dd(p.xz);
}

float ao = calcAO(p,n);
float amb = clamp(0.5 + 0.5 * n.y,0.,1.);
float dif = clamp(dot(n,l),0.0,1.0);
float spe = pow(clamp(dot(n,h),0.0,1.0),16.)
* dif * (.04 + 0.9 * pow(clamp(1. + dot(h,rd),0.,1.),5.));

float fre = pow(clamp(1. + dot(n,rd),0.0,1.0),2.0);
float ref = smoothstep(-.2,.2,r.y);

vec3 linear = vec3(0.);

dif *= shadow(p,l);
ref *= shadow(p,r);

linear += dif * vec3(h11(100.),h11(135.),h11(12.));
linear += amb * vec3(0.002,0.01,0.05);
linear += ref * vec3(4.  );
linear += fre * vec3(0.125,0.005,0.35);

col = col * linear * ao;
col += spe * vec3(1.); 
col = mix(col,c,1.-exp(-0.00001 * d.x*d.x*d.x)); 

}

return col;
}

void main() {
 
vec3 color = vec3(0.);
vec3 ro = vec3(1.,5.,10.);
vec3 ta = vec3(0.0);

for(int k = 0; k < aa; ++k) {
    for(int l = 0; l < aa; ++l) {

    vec2 o = vec2(float(l),float(k)) / float(aa) - .5;

    vec2 uv = (2. * (gl_FragCoord.xy + o) -
    resolution.xy) / resolution.y; 

    mat3 cm = camOrthographic(ro,ta,0.);
    vec3 rd = cm * normalize(vec3(uv.xy,2.));

    vec3 render = renderScene(ro,rd,vec3(10.),vec3(1.));    

    color += render;
    }

color /= float(aa*aa);
color = pow(color,vec3(.4545));

FragColor = vec4(color,1.0);
}

}
