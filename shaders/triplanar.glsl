#version 330 core

vec4 out FragColor;
uniform vec2 resolution;
uniform sampler2D tex;
uniform float time;

#define STEPS 255
#define EPS 0.001
#define NEAR 0.
#define FAR 75.

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

vec2 opu(vec2 d1,vec2 d2) {
    return (d1.x < d2.x) ? d1 : d2;
} 

float box(vec3 p,vec3 b) {

    vec3 d = abs(p) - b;
    return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0);
}


vec2 scene(vec3 p) {

    vec2 res = vec2(1.,0.);

    float d = 0.;     
    d = box(p,vec3(1.));

    res = opu(res,vec2(d,2.)); 
    res = opu(res,vec2(-box(p,vec3(25.)),3.));

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

vec3 calcNormal(vec3 p) {

    vec2 e = vec2(1.,-1.)*EPS;

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
vec3 l = normalize(vec3(10.));
l.yz *= rot(time*.01);

float dif = dot(n,l)*0.5+0.5;

vec3 col_xy = texture(tex,p.xy*0.5+0.5).rgb;
vec3 col_xz = texture(tex,p.xz*0.5+0.5).rgb;
vec3 col_zy = texture(tex,p.zy*0.5+0.5).rgb;

n = abs(n);

col = col_xy*n.z + col_xz*n.y + col_zy*n.x;

col *= dif;
col = mix(col,vec3(1.),1.-exp(-0.00001 * d.x*d.x*d.x)); 

}

return col;
}

void main() {

vec3 color = vec3(0.);

vec3 ro = vec3(2.);
ro.xy *= rot(time*0.15);

vec3 ta = vec3(0.0);
vec2 uv = (2. * gl_FragCoord.xy-resolution.xy)/resolution.y;  

mat3 cm = camOrthographic(ro,ta,0.);
vec3 rd = cm * normalize(vec3(uv,2.));

vec3 col = renderScene(ro,rd);

col = pow(col,vec3(0.4545));
color += col;

gl_FragColor = vec4(color,1.0);
}
