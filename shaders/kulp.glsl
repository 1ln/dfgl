#version 330 core     

out vec4 fragColor;

uniform vec2 resolution;
uniform float time;

uint pcg(uint p) {
    uint state = p* 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u))
    ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float plot(vec2 p,float x) {
     return smoothstep(x-.01,x,p.y)-
            smoothstep(x,x+.01,p.y);
}

void main() {
 
    vec2 p = (2.*gl_FragCoord.xy-resolution)/resolution.y;  



    vec2 q = p;
    float d = length(q);

    float a = .5;
    float y = .5*a/sqrt(p.x*p.x+a*a);
    p.y = abs(p.y); 

    float y2 = .001+cos(p.x*p.y*3.)-.5;
    


    vec3 c;

    c += 1.- smoothstep(.001,.005,d);  

    c = pow(c,vec3(.4545));
    fragColor = vec4(c,1.);

}
