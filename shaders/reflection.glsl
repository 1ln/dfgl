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

float easeOut3(float t) {
    return (t = t - 1.0) * t * t + 1.0;
}


vec3 scatter(vec3 col,vec3 tf,vec3 ts,vec3 rd,vec3 l,float de) {
    float fog_depth  = 1. - exp(-0.000001 * de);
    float light_depth = max(dot(rd,l),0.);
    vec3 fog_col = mix(tf,ts,pow(light_depth,8.));
    return mix(col,fog_col,light_depth);
}

float sphere(vec3 p,float r) { 
    return length(p) - r;
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

void main() { 
vec3 color = vec3(0.);

vec3 ta = vec3(0.);
vec3 ro = camPos;

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
