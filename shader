 







//enso
//2022
//Dan Olson

//Within imperfection is beauty 
//Even so, there is a path to perfection.

#define FC fragCoord.xy
#define RE iResolution
#define T iTime

#define EPS 0.000313
#define STEPS 75
#define NEAR 0.
#define FAR 5.

//zero for no gamma
#define GAMMA .4545

#define PI radians(180.)

float h(float p) {
    return fract(sin(p)*float(43758.98));
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
                vec3 r = vec3(h(p.x+b.x+h(p.y+b.y+h(p.z+b.z))));
                
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

    return mix(mix(mix(h(q + 0.0),h(q + 1.0),f.x),
           mix(h(q + 157.0),h(q + 158.0),f.x),f.y),
           mix(mix(h(q + 113.0),h(q + 114.0),f.x),
           mix(h(q + 270.0),h(q + 271.0),f.x),f.y),f.z);
}

float expstep(float x,float k) {
    return exp((x*k)-k);
}

vec2 opu(vec2 d1,vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
} 

float lv(vec3 p,float d,float h) {
    vec2 w = vec2(d,abs(p.z) - h);
    return min(max(w.x,w.y),0.) + length(max(w,0.)); 
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

float arc(vec2 p,vec2 sca,vec2 scb,float ra,float rb) {
    p *= mat2(sca.x,sca.y,-sca.y,sca.x);
    p.x = abs(p.x);
    float k = (scb.y*p.x>scb.x*p.y) ? dot(p,scb) : length(p);
    return sqrt(dot(p,p)+ra*ra-2.*ra*k)-rb;
}

float plane(vec3 p,vec4 n) {
    return dot(p,n.xyz) + n.w;
}

#define R res = opu(res,vec2(
vec2 scene(vec3 p) { 

vec2 res = vec2(1.0,0.0);

vec3 q = p;
R plane(q,vec4(0.,1.,0.,1.)),2.));

p.xz *= rot(T * .1);
q.xz *= rot(T * .3);

float a = 2.5;
R lv(p.xzy,arc(q.xz+n3(p+n3(p*.1) )*.5  ,vec2(.5,.5), 
vec2(sin(a),cos(a)),3.25,.5),
.001),1.));

R length(p-vec3(0.,2.,0.))-1.5,2.)); 

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
        h = 1.;   

        if(abs(dist.x) < EPS || e <  dist.x ) { break; }
        s += dist.x;
        d = dist.y;

        }

        if(e < s) { d = -1.0; }
        return vec4(s,d,h,1.);

}

float shadow(vec3 ro,vec3 rd ) {
    float res = 1.0;
    float dmax = 24.;
    float t = 0.5;
    float ph = 1e10;
    
    for(int i = 0; i < 125; i++ ) {
        
        float h = scene(ro + rd * t  ).x;

        float y = h * h / (2. * ph);
        float d = sqrt(h*h-y*y);         
        res = min(res,24. * d/max(0.,t-y));
        ph = h;
        t += h;
    
        if(res < EPS || t*rd.y+ro.y > dmax) { break; }

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

vec3 aces(vec3 x) {
    return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.);
}  

vec3 glow_exp(vec3 c,vec3 rd,vec3 l) {
    float rad = dot(rd,l);
    c += c * vec3(2.,1.,.1) * expstep(rad,250.);
    c += c * vec3(2.,.5,1.) * expstep(rad,125.);  
    c += c * vec3(1.,.1,.5) * expstep(rad,100.);       
    c += c * vec3(.5,1.,1.) * expstep(rad,75.);
    return c;
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

vec3 linear(vec3 ro,
            vec3 rd,
            vec3 n,            
            vec3 l,           
            vec4 d
            ) {

    vec3 p = ro + rd * d.x;

    vec3 linear = vec3(0.);

    float amb = ambient(n);
    float dif = diffuse(n,l);
    float spe = specular(n,rd,l);
    float fre = fresnel(n,rd);   
  
    float sh = shadow(p,l);
         
    linear += dif * vec3(.5) * sh;
    linear += amb * vec3(0.01);
    linear += 1. * fre * vec3(.1);
    linear += .95 * spe * vec3(1.,.75,.91);                

    return linear;   
    
}

vec3 render(vec3 ro,vec3 rd,vec3 l) {

    vec3 c = vec3(0.),
         fc = vec3(.5);   
    
    vec4 d;

    for(int i = 0; i < 3; i++) {
       d = trace(ro,rd);
        
       vec3 p = ro + rd * d.x;
       vec3 n = calcNormal(p);
       vec3 r = reflect(rd,n);
       c += linear(p,rd,n,l,d);
       ro = p + n * .005;  
      
       if(d.y >= 0.) {

            if(d.y == 1.) {
               c *= vec3(.5,1.,.5);
               p.xz *= rot(T*.12);           
               c *= mix(c,vec3(0.,1.,0.),n3(p+n3(p*12. ) ));           
               c += mix(c,vec3(.1),cell(p,2.));  

           }
    
           if(d.y == 2.) {
               rd = r;   
               c += vec3(.5);
           }        
       }
       fc *= mix(c,vec3(1.),1. - exp(-.5 * float(i)));
}
return fc;
}


#define AA 1
void mainImage(out vec4 fragColor,in vec2 fragCoord) { 

vec3 fc = vec3(0.);

vec3 ta = vec3(0.);
vec3 ro = vec3(2.,13.,3.);
vec3 rd = vec3(0.);
vec3 l = normalize(vec3(34.));
l.xz *= rot(T*.1);

for(int i = 0; i < AA; i++ ) {
   for(int k = 0; k < AA; k++) {
   
       vec2 o = vec2(float(i),float(k)) / float(AA) * .5;
       vec2 uv = (2.* (FC+o) -
       RE.xy)/RE.y;
 
       mat3 cm = camera(ro,ta,2.);
       rd = cm * normalize(vec3(uv.xy,3.));
       vec4 d = trace(ro,rd);            
       vec3 p = ro + rd * d.x;
       vec3 n = calcNormal(p);
       vec3 c = render(ro,rd,l);

       c *= c + glow_exp(c,rd,l);  

       c = aces(c);
     
       c = pow(c,vec3(GAMMA));
       fc += c;
   }
}   
  

   fc /= float(AA*AA);

   fragColor = vec4(fc,1.0);


}
