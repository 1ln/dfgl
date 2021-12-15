#version 330 core     

out vec4 fragColor;

uniform vec2 resolution;
uniform float time;
 
float plot(vec2 p,float x) {
     return smoothstep(x-.01,x,p.y)-
            smoothstep(x,x+.01,p.y);
}

void main() {
 
    vec2 p = (2.*gl_FragCoord.xy-resolution.xy)/resolution.y;  

    float d = length(p) +.025;

    float a = .5;
    
    //kulp quartic
    float y = .5*a/sqrt(p.x*p.x+a*a);
    
    p.y = abs(p.y); 

    float y2 = .001+cos(p.x*p.y*3.)-.5;
    float r = plot(p,y);
    float b = plot(p,y2);

    vec3 c = vec3(.1);  
    c = mix(c,vec3(0.,1.,0.),1.-smoothstep(.48,.5,d));
    c += mix(c,vec3(0.,0.,1.),smoothstep(.1,.5,b));
    c += mix(c,vec3(1.,0.,0.),smoothstep(.3,.5,r));

    fragColor = vec4(c,1.);

}
