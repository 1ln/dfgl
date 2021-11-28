#version 330     
out vec4 FragColor; 
uniform vec2 resolution;
uniform float time;
#define fragCoord gl_FragCoord

//solid angle reflections
//2021
//do

const int seed = 19222;

#define EPS 0.0001
#define PI2 radians(180.)*2.

float h11(float p) {
    return fract(sin(p)*float(43758.5453+seed));
}

float h21(vec2 p) {
    return fract(sin(dot(p,vec2(12.9898,78.233)*float(43758.5453+seed))));
}

float checkerboard(vec3 p,float h) {
    vec3 q = floor(p*h);
    return mod(q.x+q.z,2.);
}

float expStep(float x,float k) {
    return exp((x*k)-k);
}

vec2 boxBound(vec3 ro,vec3 rd,vec3 rad) {
    vec3 m =  1./rd;
    vec3 n = m * ro;
    vec3 k = abs(m) * rad;
    vec3 t1 = -n - k;
    vec3 t2 = -n + k;
    return vec2(max(max(t1.x,t1.y),t1.z),
                min(min(t2.x,t2.y),t2.z));
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

float b,b1,b2;
b = box(q-vec3(-3.,1.,-5.5),vec3(1.));
b1 = box(q-vec3(-4.,2.,-5.),vec3(2.));
b2 = box(q-vec3(-5.,.5,5.),vec3(.5));

res = opu(res,vec2(b,2.));
res = opu(res,vec2(b1,3.));
res = opu(res,vec2(b2,4.));

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
    
        if(res < EPS || t > 2.) { break; }

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
    vec3 bcol = vec3(1.);
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

        dif *= shadow(p,l);

        linear += dif * vec3(1.);
        linear += amb * vec3(0.5);
        linear += fre * vec3(.025,.01,.03);
        linear += spe * vec3(0.04,0.05,.05);

        if(d.y == 5.) {
            col = vec3(.5);
            ref = vec3(.5);
            //rd = r;                 
        }

        if(d.y == 2.) {
            col = vec3(1.,0.,0.);
            ref = vec3(0.1);   
        }    
  
        if(d.y == 3.) {
            col = vec3(0.,1.,0.);
            ref = vec3(.1);
        }

        if(d.y == 4.) {
            col = vec3(0.,0.,1.);
            ref = vec3(.1);
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

vec3 ta = vec3(0.5);
vec3 ro = vec3(-2.,2.,-1.3);

       vec2 uv = (2.* (gl_FragCoord.xy) 
       - resolution.xy)/resolution.y;

       vec3 rd = rayCamDir(uv,ro,ta,2.); 
       vec3 ref = vec3(0.);
       vec3 col = render(ro,rd,ref);       
       vec3 c = vec3(.5); 
       col += c * render(ro,rd,ref);
       col = pow(col,vec3(.4545));
       color += col;
       FragColor = vec4(color,1.0);
 

}
