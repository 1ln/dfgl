#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp> 

#include "Shader.h"

#include <iostream>

void framebuffer_size_callback(GLFWwindow* window,int w,int h);
void mouse_callback(GLFWwindow* window,double xpos,double ypos);
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

bool key_w = false;
bool key_s = false;
bool key_a = false;
bool key_d = false;
bool key_space = false; 

int main() {
 
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width,height,"dfgl",NULL,NULL);

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

    glfwSetInputMode(window,GLFW_CURSOR,GLFW_CURSOR_DISABLED);

    //glEnable(GL_DEPTH_TEST); 

    Shader shader("vert.vert","render.frag");
    
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

    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,3*sizeof(float),(void*)0);
    glEnableVertexAttribArray(0);

    unsigned int fb = 0;
    glGenFramebuffers(1,&fb);
    glBindFramebuffer(GL_FRAMEBUFFER,fb);

    unsigned int tex;
    glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D,tex);

    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,width,height,0,GL_RGB,
    GL_UNSIGNED_BYTE,0);

    glTexParameter(GL_TEXTURE,GL_TEXTURE_MAG_FILTER,GL_NEAREST); 
    glTexParameter(GL_TEXTURE,GL_TEXTURE_MIN_FILTER,GL_NEAREST);

    

    shader.use();

    while (!glfwWindowShouldClose(window)) {
  
        float current_frame = glfwGetTime(); 
        dt = current_frame - last_frame;
        last_frame = current_frame;

        processInput(window);

        glClearColor(0.5f,0.0f,0.0f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        shader.use();

        glm::vec2 resolution = glm::vec2(width,height);
        shader.setVec2("resolution",1,resolution);

        shader.setFloat("time",last_frame);   

        shader.setVec2("mouse",mouse);
        shader.setBool("mouse_pressed");

        shader.setBool("key_space",key_space);
        shader.setBool("key_w",key_w);
        shader.setBool("key_s",key_s);
        shader.setBool("key_a",key_a);       
        shader.setBool("key_d",key_d);

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

    if(glfwGetKey(window,GLFW_KEY_W) == GLFW_PRESS)
        key_w = true;

    if(glfwGetKey(window,GLFW_KEY_A) == GLFW_PRESS)
        key_s = true;       

    if(glfwGetKey(window,GLFW_KEY_S) == GLFW_PRESS)
        key_a = true;

    if(glfwGetKey(window,GLFW_KEY_D) == GLFW_PRESS)
        key_d = true;


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

    mouse = glm::vec2(lastx,lasty); 

}

void scroll_callback(GLFWwindow* window,double xoff, double yoff) {
    mouse_scroll -= float(yoff);
}

