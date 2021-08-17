#version 330 core     

//Dan Olson
//2021

out vec4 FragColor;

uniform vec2 resolution;
uniform float time;

float h11(float p) {
    uvec2 n = uint(int(p)) * uvec2(uint(int(254446)),2531151992.0);
    uint h = (n.x ^ n.y) * uint(int(254446));
    return float(h) * (1./float(0xffffffffU));
}

mat2 rot(float a) {

    float c = cos(a);
    float s = sin(a);
    
    return mat2(c,-s,s,c);
}

void main() {

vec2 uv = (gl_FragCoord.xy - 0.5 * resolution.xy)/resolution.y;

vec3 color = vec3(0.);
vec3 ro = vec3(0.,0.,-1.);
ro = vec3(0.,0.,-1.);
vec3 ta = vec3(0.0);
float fov = 0.005 + abs(sin(0.005))/0.012; 

vec3 w = normalize(ta-ro),
     r = normalize(cross(vec3(0.,1.,0.),w)),
     u = cross(w,r),
     v = vec3(ro + w * fov) + uv.x * r + uv.y * u,
     rd = normalize(v-ro);
 
vec3 p;
float dist = 0.;
float d = 0.;

    for(int i = 0; i < 350; i++) {

         p = ro + rd * dist;

         p.xz *= rot(time*.1);
         p.yz *= rot(sin(time*.5)*.15);

         d = -(length(vec2(length(p.xz)-1.,p.y)) - 0.5);

         if(d < 0.001) break;
         dist += d;

    }

if(d < 0.001) {
 
    float x = atan(p.x,p.z)*0.5;
    float y = atan(length(p.xz) - 1.,p.y);
    float b = sin(x*12. + y*10.);

    float n = smoothstep(-0.5,0.5,p.z);
    float e = smoothstep(-1.,1.,reflect(p.z+b*p.y,b*x*y));
    color += smoothstep(-2.25,2.25,n+e);
}



FragColor = vec4(pow(color,vec3(0.4545)),1.0);

}
