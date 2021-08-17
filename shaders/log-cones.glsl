#version 330 core

//dan olson

out vec4 FragColor;

uniform vec2 resolution;
uniform float time;
uniform int seed;

const int steps = 250;
const float eps = 0.0001;
const float far = 500.;
const float near = .1;

float hash(float p) {

    uvec2 n = uint(int(p)) * uvec2(1391674541U,2531151992.0 * float(seed));
    uint h = (n.x ^ n.y) * 1391674541U;
    return float(h) * (1.0/float(0xffffffffU));

}

float ns(vec3 x) {

    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 157.0 + 113.0 * p.z;

    return mix(mix(mix(hash(  n +   0.0) , hash(   n +   1.0)  ,f.x),
    mix(hash(  n + 157.0) , hash(   n + 158.0)   ,f.x),f.y),
    mix(mix(hash(  n + 113.0) , hash(   n + 114.0)   ,f.x),
    mix(hash(  n + 270.0) , hash(   n + 271.0)   ,f.x),f.y),f.z);
}

float f(vec3 x,int octaves,float h) {

    float t = 0.0;

    float g = exp2(-h); 

    float a = 0.5;
    float f = 1.0;

    for(int i = 0; i < octaves; i++) {
 
    t += a * ns(f * x); 
    f *= 2.0; 
    a *=  g;  
    
    }    

    return t;
}

vec2 opu(vec2 d1,vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
} 

float sphere(vec3 p,float r) {
    return length(p) - r;
}

float roundCone(vec3 p,float r1,float r2,float h) {

    vec2 q = vec2(length(vec2(p.x,p.y)),p.z);
    float b = (r1-r2)/h;
    float a = sqrt(1.0 - b*b);
    float k = dot(q,vec2(-b,a));

    if( k < 0.0) return length(q) - r1;
    if( k > a*h) return length(q - vec2(0.0,h)) - r2;

    return dot(q,vec2(a,b)) - r1;
} 

vec2 scene(vec3 p) { 

vec2 res = vec2(1.0,0.0);
float scale = float(45.) / radians(180.);

vec2 h = p.xz; 
float r = length(h); 
h = vec2(log(r),atan(h.y,h.x));
h *= scale;
h = mod(h,2.) - 1.;
float mul = r/scale;

p.y += ns(p * .5) + .5;

float d = 0.;
d = roundCone(vec3(h,p.y/mul),1.,.5,2.) * mul; 

res = vec2(d,1.);

return res;

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

vec3 rayCamDir(vec2 uv,vec3 camPosition,vec3 camTarget,float fPersp) {

     vec3 camForward = normalize(camTarget - camPosition);
     vec3 camRight = normalize(cross(vec3(0.0,1.0,0.0),camForward));
     vec3 camUp = normalize(cross(camForward,camRight));

     vec3 vDir = normalize(uv.x * 
     camRight + uv.y * camUp + camForward * fPersp);  

     return vDir;
}

vec3 render(vec3 ro,vec3 rd,float dist) {

vec3 col = vec3(.5);
vec3 bgcol = vec3(1.) * max(0.,rd.y);

ro += rd * (dist*.98); 
vec3 p = ro;

vec3 n = calcNormal(p);
vec3 l = normalize(vec3(2.,10.,1.));
vec3 h = normalize(l - rd);
vec3 r = reflect(rd,n);

float amb = sqrt(clamp(0.5 + 0.5 * n.y,0.0,1.0));
float dif = clamp(dot(n,l),0.0,1.0);
float spe = pow(clamp(dot(n,h),0.0,1.0),16.) * dif * (.04 + 0.9 * pow(clamp(1. + dot(h,rd),0.,1.),5.));

vec3 linear = vec3(1.);

linear += dif * vec3(.15);
linear += amb * vec3(.03);

col = col * linear;
col += 5. * spe * vec3(.5);
col = mix(col,bgcol,1.-exp(.005*dist*dist));

return col;
}

void main() {

vec3 ta = vec3(0.);
vec3 ro = vec3(1.);
ro = ta + vec3(cos(time*.1)-4.,2.,sin(time*.25)+3.);

vec2 uv = (2. * gl_FragCoord.xy - resolution) / resolution.y;

float fov = 2.;
float vfov = 1.;

vec3 color = vec3(1.);
float dist = eps; 
float d = near;

float radius = 2. * tan(vfov/2.) / resolution.y * 1.5;
vec3 rd = rayCamDir(uv,ro,ta,fov);

vec4 col_alpha = vec4(0.,0.,0.,1.);
 
for(int i = 0; i < steps; i++ ) {
    float rad = d * radius;
    dist = scene(ro + d * rd).x;
 
    if(dist < rad) {
        float alpha = smoothstep(rad,-rad,dist);
        vec3 col = render(ro,rd,d);
        col_alpha.rgb += col_alpha.a * (alpha * col.rgb);
        col_alpha.a *= (1. - alpha);

        if(col_alpha.a < eps) break;
    
    }

    d += max(abs(dist * .75 ), .001);
    if(d > far) break;
}

color = mix(col_alpha.rgb,color,col_alpha.a);

FragColor = vec4(pow(color,vec3(.4545)),1.0);
 
}
