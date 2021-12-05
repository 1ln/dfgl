#version 330 core     

out vec4 fragColor;

uniform vec2 resolution;
uniform float time;

mat2 rot2(float a) {

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

float smou(float d1,float d2,float k) {

    float h = clamp(0.5 + 0.5 * (d2-d1)/k,0.0,1.0);
    return mix(d2,d1,h) - k * h * (1.0 - h);
}

float sphere(vec3 p,float r) { 
    return length(p)-r;
}

float scene(vec3 p) {

    float d = 0.;     
    float r = 8.;  
    
    d = sphere(p,r);

    for(int i = 0; i < 3; i++) {

        p.xz *= rot2(radians(180.)/2. * time * 0.01);
        p.zy *= rot2(radians(180.));
        p.xy *= rot2(radians(180.)*2. - 1.);

        p = abs(p) - r * 0.5;
        r /= 3.;
  
        d = smou(d,sphere(p,r),0.1);

    }

    return d;

}

float calcAO(vec3 p,vec3 n) {

    float o = 0.;
    float s = 1.;

    for(int i = 0; i < 15; i++) {
 
        float h = .01 + .125 * float(i) / 4.; 
        float d = scene(p + h * n);  
        o += (h-d) * s;
        s *= .9;
        if(o > .33) break;
    
     }
     return clamp(1. - 3. * o ,0.0,1.0) * (.5+.5*n.y);   
}

vec3 calcNormal(vec3 p) {

    vec2 e = vec2(1.0,-1.0) * 0.0001;

    return normalize(vec3(
    vec3(e.x,e.y,e.y) * scene(p + vec3(e.x,e.y,e.y)) +
    vec3(e.y,e.x,e.y) * scene(p + vec3(e.y,e.x,e.y)) +
    vec3(e.y,e.y,e.x) * scene(p + vec3(e.y,e.y,e.x)) + 
    vec3(e.x,e.x,e.x) * scene(p + vec3(e.x,e.x,e.x))

    ));
    
}

void main() {

vec3 color = vec3(0.);
vec3 ro = vec3(10.,5.,25.);
vec3 ta = vec3(0.0);

vec3 n;
vec3 l = normalize(vec3(10.));
vec3 linear = vec3(0.); 

vec2 uv = (2. * (gl_FragCoord.xy) -
resolution.xy) / resolution.y; 

mat3 cm = camOrthographic(ro,ta,0.);

vec3 rd = cm * normalize(vec3(uv.xy,2.));

float dist = 0.;

    for(int i = 0; i < 350; i++) {

         vec3 p = ro + rd * dist;
         float d = scene(p);

         if(d < 0.001) {

             n = calcNormal(p);
 
             float amb = clamp(0.5+0.5*n.y,0.,1.);

             float dif = clamp(dot(
             n,l),0.,1.);

             linear += dif * vec3(float(i)/100.,0.,0.); 
             linear += amb * vec3(0.5);
             float ao = calcAO(p*1.1,n);
             color += linear*ao;

             break;
           
         }
         dist += d;
         color = vec3(1.,
         smoothstep(-1.,1.,rd.x),smoothstep(-1.,1.,rd.y));
    }

fragColor = vec4(pow(color,vec3(0.4545)),1.0);

}
