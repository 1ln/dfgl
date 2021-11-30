#version 330     
out vec4 FragColor; 
uniform vec2 resolution;
uniform float time;

#define fragCoord gl_FragCoord
#define R resolution
#define t time

//convexity
//2021
//do

const int seed = 19222;

#define AA 2
#define EPS 0.001

float h11(float p) {
    return fract(sin(p)*float(43758.5453+seed));
}

float checkerboard(vec3 p,float h) {
    vec3 q = floor(p*h);
    return mod(q.x+q.z,2.);
}

float expStep(float x,float k) {
    return exp((x*k)-k);
}

vec2 opu(vec2 d1,vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
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

float easeOut3(float t) {
    return (t = t - 1.0) * t * t + 1.0;
}

float sphere(vec3 p,float r) { 
    return length(p) - r;
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

float torus(vec3 p,vec2 t) {
    vec2 q = vec2(length(vec2(p.x,p.z)) - t.x,p.y);
    return length(q) - t.y; 
}

float cylinder(vec3 p,float h,float r) {
    vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
    return min(max(d.x,d.y),0.) + length(max(d,0.));
}

float tetrahedron(vec3 p,float h) {
     vec3 q = abs(p);
     float y = p.y;
     float d1 = q.z-max(y,0.);
     float d2 = max(q.x*.5+y*.5,0.)-min(h,h+y);
     return length(max(vec2(d1,d2),.005)) + min(max(d1,d2),0.);
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

float box(vec3 p,vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0);
}

float sa(vec3 p,vec2 c,float ra) {
    vec2 q = vec2(length(p.xy),p.z);
    float l = length(q) - ra;
    float m = length(q - c * clamp(dot(q,c),0.,ra));
    return max(l,m * sign(q.x * c.y - q.y * c.x));
}

vec2 scene(vec3 p) { 

vec2 res = vec2(1.0,0.0);
vec3 q = p;

float b;
b = box(q-vec3(3.,1.,-6.),vec3(1.));
res = opu(res,vec2(b,2.));

res = opu(res,vec2(sphere(q-vec3(-6.,2.,3.),2.),39.453));

res = opu(res,vec2(
capsule(q-vec3(-7.,.5,.9)
,vec3(-2.,3.,-1.),vec3(0.),.5),225.9));

res = opu(res,vec2(torus(q.yzx-vec3(1.5,-3.5,-5.5),vec2(1.,.5)),12.6));
res = opu(res,vec2(tetrahedron(q-vec3(-3.,.1,-5.),.75),75.67));
res = opu(res,vec2(octahedron(q-vec3(-.75,1.,-5.1),1.),100.1));
res = opu(res,vec2(cylinder(q-vec3(-5.,1.,-1.),.5,.25),64.364));
res = opu(res,vec2(prism(q-vec3(-5.5,1.,-2.),vec2(1.,.25)),124.5));

p.xz *= rot(-.5+easeOut3(sin(.5*t)*.25)+1.25);
vec3 n = p;
n.zy *= rot(-2.5);

float sc = sa(n,vec2(.6,.8),1.);
res = opu(res,vec2(sc,5.));

res = opu(res,vec2(p.y,1.));

return res;


}

vec2 trace(vec3 ro,vec3 rd) {
    
    float d = -1.0;
    float s = 0.;
    float e = 16.;  

    for(int i = 0; i < 75; i++) {

        vec3 p = ro + s * rd;
        vec2 dist = scene(p);
   
        if(dist.x < EPS || e <  dist.x ) { break; }
        s += dist.x;
        d = dist.y;

        }
 
        if(e < s) { d = -1.0; }
        return vec2(s,d);

}

float reflection(vec3 ro,vec3 rd ) {

    float depth = 0.;
    float dmax = 10.;
    float d = -1.0;

    for(int i = 0; i < 25; i++ ) {
        float h = scene(ro + rd * depth).x;

        if(h < .01) { return depth; }
        
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
    
        if(res < .01 || t > 12.) { break; }

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
    vec3 bcol = vec3(1.3);
    vec3 col = bcol * max(1.,rd.y);

    vec3 l = normalize(vec3(25.,3.,-35.));

    float rad = dot(rd,l);
    col += col * vec3(.5,.7,.5) * expStep(rad,100.);
    col += col * vec3(.5,.1,.15) * expStep(rad,125.);
    col += col * vec3(.1,.5,.05) * expStep(rad,25.);
    col += col * vec3(.15) * expStep(rad,35.);

    vec3 h = normalize(l - rd);  
    float dif = clamp(dot(n,l),0.0,1.0);
    float spe = pow(clamp(dot(n,h),0.0,1.0),16.)
    * dif * (.04 + 0.9 * pow(clamp(1. + dot(h,rd),0.,1.),5.));

    if(d.y >= 0.) {
        
        col = .5+.5*sin(2.*d.y+vec3(6.*h11(5.),2.,3.));

        dif *= shadow(p,l);

        linear += dif * vec3(1.9);
        linear += amb * vec3(0.5);
        linear += fre * vec3(.25,.1,.03);
        linear += spe * vec3(0.04,0.05,.05);

        if(d.y == 5.) {
            col = vec3(.5);
            ref = vec3(.5);
            //rd = r;                 
        }

        if(d.y == 1.) {
            col = vec3(checkerboard(p,1.))*vec3(1.,.5,.25);
            ref = vec3(.5);
        }

        rd = r;

        col = col * linear;
        col = mix(col,bcol,1.-exp(-.001*d.x*d.x*d.x));

}

return col;
}

void main() { 
vec3 color = vec3(0.);

vec3 ta = vec3(0.);
vec3 ro = vec3(-2.,2.,-1.3);

for(int k = 0; k < AA; k++ ) {
   for(int l = 0; l < AA; l++) {
   
       vec2 o = vec2(float(k),float(l)) / float(AA) * .5;
       vec2 uv = (2.* (gl_FragCoord.xy+o) -
       resolution.xy)/resolution.y;

       vec3 rd = rayCamDir(uv,ro,ta,2.); 
       vec3 ref = vec3(0.);
       vec3 col = render(ro,rd,ref);       
       vec3 c = vec3(.5);
       col += c * render(ro,rd,ref);
       col = pow(col,vec3(.4545));
       color += col;
   }
}
   
   color /= float(AA*AA);
   FragColor = vec4(color,1.0);
 

}
