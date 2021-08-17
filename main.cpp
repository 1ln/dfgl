#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp> 
#include "Shader.h"

#include <iostream>
#include <vector>
#include <string>

void framebuffer_size_callback(GLFWwindow* window,int w,int h);
void mouse_callback(GLFWwindow* window,double xpos,double ypos);
void mouse_button_callback(GLFWwindow* window,int button,int action,int mods); 
void scroll_callback(GLFWwindow* window,double xoff,double yoff); 
void processInput(GLFWwindow *window);

const unsigned int width  = 800;
const unsigned int height = 600;

float lastx = width / 2.0;
float lasty = height / 2.0;

bool init_mouse = true;

float dt         = 0.0f;
float last_frame = 0.0f;

glm::vec2 mouse;
bool mouse_pressed = false; 
float mouse_scroll = 0.0f;

bool key_space = false;
bool key_x = false;
bool key_z = false;
bool key_up = false;
bool key_down = false;
bool key_right = false;
bool key_left = false;

bool hide_cursor = false;

int main(int argc,char** argv) {
 
    std::string frag0 = argv[1];    
    std::string frag1 = argv[2];  

    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width,height,
    "dfgl",NULL,NULL);

    if(window == NULL) {
        std::cout << "Failed to initalize GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
   
    glewExperimental = GL_TRUE;
    glewInit();

    glfwSetFramebufferSizeCallback(window,framebuffer_size_callback);
    glfwSetCursorPosCallback(window,mouse_callback);
    glfwSetScrollCallback(window,scroll_callback);
    glfwSetMouseButtonCallback(window,mouse_button_callback);

    if(hide_cursor) {
    glfwSetInputMode(window,GLFW_CURSOR,GLFW_CURSOR_DISABLED);
    } 

    GLuint fb = 0;
    glGenFramebuffers(1,&fb);
    glBindFramebuffer(GL_FRAMEBUFFER,fb);

    GLuint tex;
    glGenTextures(1,&fb);
    glBindTexture(GL_TEXTURE_2D,tex); 

    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,width,height,0,GL_RGB,
    GL_UNSIGNED_BYTE,0);    

    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);

    glFramebufferTexture(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,tex,0);
    GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1,DrawBuffers);    

    glBindFramebuffer(GL_FRAMEBUFFER,fb);
      
    float verts[] = {
    -1.,3.,0.,
    -1.,-1.,0.,
     3.,-1.,0.
    };

    unsigned int vbo,vao;

    glGenVertexArrays(1,&vao);
    glGenBuffers(1,&vbo);
    glBindVertexArray(vao);
    
    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glBufferData(GL_ARRAY_BUFFER,sizeof(verts),verts,GL_STATIC_DRAW);

    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE 
    ,3*sizeof(float),(void*)0);

    glEnableVertexAttribArray(0);

    Shader buffer("vert.glsl",frag0.c_str());
    Shader shader("vert.glsl",frag1.c_str());         
    
    buffer.use();

    shader.use();
    
    while (!glfwWindowShouldClose(window)) {
  
        float current_frame = glfwGetTime(); 
        dt = current_frame - last_frame;
        last_frame = current_frame;

        processInput(window);

        glm::vec2 resolution = glm::vec2(width,height);        

        glBindFramebuffer(GL_FRAMEBUFFER,fb);
        glViewport(0,0,width,height);
        buffer.use();        
        buffer.setVec2("resolution",1,resolution);
        buffer.setFloat("time",last_frame);

        glBindFramebuffer(GL_FRAMEBUFFER,0);        
        glViewport(0,0,width,height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        shader.use();
        shader.setVec2("resolution",1,resolution);
        shader.setFloat("time",last_frame);
        shader.setVec2("mouse",1,mouse);
        shader.setFloat("mouse_scroll",mouse_scroll);
        shader.setBool("mouse_pressed",mouse_pressed);
        shader.setTex("tex",tex);

        glBindVertexArray(vao);

        glDrawArrays(GL_TRIANGLES,0,3);

        glfwSwapBuffers(window);
        glfwPollEvents();

    }

    glDeleteVertexArrays(0,&vao);
    glDeleteBuffers(0,&vbo);

    glfwTerminate();
    return 0;

}

void processInput(GLFWwindow *window) {

    if(glfwGetKey(window,GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window,true);



    if(glfwGetKey(window,GLFW_KEY_UP) == GLFW_PRESS) {
        key_up = true;
    } else {
        key_up = false;
    }

    if(glfwGetKey(window,GLFW_KEY_SPACE) == GLFW_PRESS) { 
        key_space = true;
    } else { 
        key_space = false;
    }
  



}

void framebuffer_size_callback(GLFWwindow* window,int w,int h) {
    glViewport(0,0,w,h);
}

void mouse_callback(GLFWwindow* window,double xpos,double ypos) { 

     if(init_mouse) {

         lastx = xpos;
         lasty = ypos; 
         init_mouse = false;

     }

    float xoff = xpos - lastx;
    float yoff = lasty - ypos;

    lastx = xpos;
    lasty = ypos;
 
    if(mouse_pressed) {
        mouse.x = lastx;
        mouse.y = lasty; 
    }
}

void mouse_button_callback(GLFWwindow* window,int button,int action,int mods) {
  
    if(button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        mouse_pressed = true;
    } else {
        mouse_pressed = false;
    }

}

void scroll_callback(GLFWwindow* window,double xoff, double yoff) {
    mouse_scroll -= float(yoff);
}

