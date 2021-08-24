const vert = `
#version 300 es

void main() {

gl_Position = vec4(position,1.);

}
`;

const frag = `

#version 300 es     

// dolson,2019

out vec4 out_FragColor;

uniform float time;
uniform vec2 res;
uniform int seed;

const float PI   =  radians(180.0); 
float eps = 0.00001;

float hash(float p) {
    uvec2 n = uint(int(p)) * uvec2(uint(int(seed)),2531151992.0);
    uint h = (n.x ^ n.y) * uint(int(seed));
    return float(h) * (1./float(0xffffffffU));
}

float hash(vec2 p) {
    uvec2 n = uvec2(ivec2(p)) * uvec2(uint(int(seed)),2531151992.0);
    uint h = (n.x ^ n.y) * uint(int(seed));
    return float(h) * (1./float(0xffffffffU));
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

float n3(vec3 x) {

    vec3 p = floor(x);
    vec3 f = fract(x);

    f = f * f * (3.0 - 2.0 * f);
    float n = p.x + p.y * 157.0 + 113.0 * p.z;

    return mix(mix(mix(hash(  n +   0.0) , hash(   n +   1.0)  ,f.x),
                   mix(hash(  n + 157.0) , hash(   n + 158.0)   ,f.x),f.y),
               mix(mix(hash(  n + 113.0) , hash(   n + 114.0)   ,f.x),
                   mix(hash(  n + 270.0) , hash(   n + 271.0)   ,f.x),f.y),f.z);
}

float f3(vec3 x,float hurst) {

    float s = 0.;
    float h = exp2(-0.502);
    float f = 1.;
    float a = .5;

    for(int i = 0; i < 4; i++) {

        s += a * n3(f * x);  
        f *= 2.;
        a *= h;
    }
    return s;
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

float ls(float s,float e,float t) {
    return clamp((t-s)/(e-s),0.,1.);
}

mat2 rot2(float a) {

    float c = cos(a);
    float s = sin(a);
    
    return mat2(c,-s,s,c);
}

mat4 rotAxis(vec3 axis,float theta) {

axis = normalize(axis);

    float c = cos(theta);
    float s = sin(theta);

    float oc = 1.0 - c;

    return mat4(
 
        oc * axis.x * axis.x + c, 
        oc * axis.x * axis.y - axis.z * s,
        oc * axis.z * axis.x + axis.y * s, 
        0.0,
        oc * axis.x * axis.y + axis.z * s,
        oc * axis.y * axis.y + c, 
        oc * axis.y * axis.z - axis.x * s,
        0.0,
        oc * axis.z * axis.x - axis.y * s,
        oc * axis.y * axis.z + axis.x * s, 
        oc * axis.z * axis.z + c, 
        0.0,
        0.0,0.0,0.0,1.0);

}

vec2 opu(vec2 d1,vec2 d2) {

    return (d1.x < d2.x) ? d1 : d2;
} 

float plane(vec3 p,vec4 n) {

    return dot(p,n.xyz) + n.w;
}

float box(vec3 p,vec3 b) {

    vec3 d = abs(p) - b;
    return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0);
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
  
    float s = 0.0001;
    float t = time;  
    
    vec3 q=p,l=p;

    p.xz *= rot2(t*s   );

    float ft = mod(time*0.001,4.);

    float e1,e2,e3,e4;
    e1 = easeIn4(ls(0.,1.,ft));
    e2 = easeInOut4(ls(1.,2.,ft)); 
    e3 = easeOut3(ls(2.,1.,ft));
    e4 = easeInOut3(ls(1.,0.,ft));

    d = max(
    -plane(p*.5,vec4(1.,-1.,-1.,e1+e2+e3+e4-2.)),
    menger(p,4,1.,box(p,vec3(1.)))); 
    res = opu(res,vec2(d,2.)); 

    float pl = plane(l,vec4(0.,1.,0.,1.));
    res = opu(res,vec2(pl,1.));
  
    return res;

}

vec2 rayScene(vec3 ro,vec3 rd) {
    
    float d = -1.0;
    float s = 0.;
    float e = 100.;  

    for(int i = 0; i < 255; i++) {

        vec3 p = ro + s * rd;
        vec2 dist = scene(p);
   
        if(abs(dist.x) < eps || e <  dist.x ) { break; }
        s += dist.x;
        d = dist.y;

        }
 
        if(e < s) { d = -1.0; }
        return vec2(s,d);

}

float shadow(vec3 ro,vec3 rd ) {

    float res = 1.0;
    float t = 0.005;
    float ph = 1e10;
    
    for(int i = 0; i < 100 ; i++ ) {
        
        float h = scene(ro + rd * t  ).x;

        float y = h * h / (2. * ph);
        float d = sqrt(h*h-y*y);         
        res = min(res,45. * d/max(0.,t-y));
        ph = h;
        t += h;
    
        if(res < eps || t > 225.) { break; }

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

vec3 rayCamDir(vec2 uv,vec3 camPosition,vec3 camTarget,float fPersp) {

     vec3 camForward = normalize(camTarget - camPosition);
     vec3 camRight = normalize(cross(vec3(0.0,1.0,0.0),camForward));
     vec3 camUp = normalize(cross(camForward,camRight));

     vec3 vDir = normalize(uv.x * camRight + uv.y 
     * camUp + camForward * fPersp);  

     return vDir;
}

vec3 render(vec3 ro,vec3 rd) {
 
vec2 d = rayScene(ro, rd);

vec3 col = vec3(1.) - max(rd.y,0.);

if(d.y >= 0.) { 

vec3 p = ro + rd * d.x;
vec3 n = calcNormal(p);
vec3 l = normalize(vec3(0.,10.,10.));
vec3 h = normalize(l - rd);
vec3 r = reflect(rd,n);

float amb = sqrt(clamp(0.5 + 0.5 * n.y,0.0,1.0));
float dif = clamp(dot(n,l),0.0,1.0);

float spe = pow(clamp(dot(n,h),0.0,1.0),16.)
* dif * (.04 + 0.9 * pow(clamp(1. + dot(h,rd),0.,1.),5.));

float fre = pow(clamp(1. + dot(n,rd),0.0,1.0),2.0);
float ref = smoothstep(-.2,.2,r.y);

vec3 linear = vec3(0.);

dif *= shadow(p,l);
ref *= shadow(p,r);

linear += dif * vec3(.5);
linear += amb * vec3(.05);
linear += ref * vec3(1.,0.34,0.5);
linear += fre * vec3(0.05,0.005,0.1);

if(d.y == 1.) { 
col = vec3(0.5,0.34,0.24);
}

if(d.y == 2.) { 
col = vec3(0.5);                        
}

col = col * linear;
col += 5. * spe * vec3(hash(547.));

col = mix(col,vec3(1.),1. - exp(-.0001 * d.x * d.x * d.x));

} 

return col;
}

void main() {
 
vec3 color = vec3(0.);

vec3 cam_tar = vec3(0.);
vec3 cam_pos = vec3(3.);

vec2 uv = (2.*gl_FragCoord.xy-res.xy)/res.y; 

vec3 dir = rayCamDir(uv,cam_pos,cam_tar,2.); 

color = render(cam_pos,dir);  

color = pow(color,vec3(.4545));
out_FragColor = vec4(color,1.0);

}
`;

let renderer,canvas,context;

let uniforms;

let scene,material,mesh;

let plane,w,h; 

let cam;
let r = new Math.seedrandom();
let s = Math.abs(r.int32());

init();
render();

function init() {

    canvas = $('#canvas')[0];
    context = canvas.getContext('webgl2');
    
    w = window.innerWidth;
    h = window.innerHeight;

    canvas.width = w;
    canvas.height = h;

    scene = new THREE.Scene();
    plane = new THREE.PlaneBufferGeometry(2,2);

    renderer = new THREE.WebGLRenderer({
    
        canvas : canvas,
        context : context

    });

    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(w,h);
    
    cam = new THREE.PerspectiveCamera(0.,w/h,0.,1.);

    material = new THREE.ShaderMaterial({

       uniforms : {
    
        
           res        : new THREE.Uniform(new THREE.Vector2(w,h)),
           time          : { value : 1. },
           seed          : { value : s }

       },

       vertexShader   : vert,
       fragmentShader : frag

    });

    mesh = new THREE.Mesh(plane,material);
    scene.add(mesh);

}

    function render() {
 
    material.uniforms.res.value = new THREE.Vector2(w,h);
    material.uniforms.time.value = performance.now();

    renderer.render(scene,cam);
    requestAnimationFrame(render);

}

