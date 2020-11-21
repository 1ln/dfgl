#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp> 
#include <glm/gtx/noise.hpp>

#include <gli/gli.hpp>

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

bool key_w = false;
bool key_a = false;
bool key_s = false;
bool key_d = false;

bool key_up = false;
bool key_down = false;
bool key_right = false;
bool key_left = false;

bool key_q = false;
bool key_e = false; 

bool key_space = false;

int seed = 1251623;

int main(int argc,char** argv) {
 
    std::string frag = argv[1];    

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

    //glfwSetInputMode(window,GLFW_CURSOR,GLFW_CURSOR_DISABLED);

    Shader shader("vert.vert",frag.c_str());

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

    unsigned int fb = 0;
    glGenFramebuffers(1,&fb);
    glBindFramebuffer(GL_FRAMEBUFFER,fb);

    unsigned int tex;
    glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D,tex);

    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB,width,height,0,GL_RGB,
    GL_UNSIGNED_BYTE,NULL);

    glTexParameteri(GL_TEXTURE,GL_TEXTURE_MAG_FILTER,GL_NEAREST); 
    glTexParameteri(GL_TEXTURE,GL_TEXTURE_MIN_FILTER,GL_NEAREST);

    glFramebufferTexture(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,
                         tex,0);

    GLenum draw_buffers[1] = { GL_COLOR_ATTACHMENT0 }; 
    glDrawBuffers(1,draw_buffers);  

    unsigned int ntex;
    glGenTextures(1,&ntex);
    glBindTexture(GL_TEXTURE_2D,ntex);

    std::vector<GLfloat> image(3*width*height);

    for(int j = 0; j < height; ++j) {
        for(int i = 0; i < width; ++i) {
    
        size_t ind = j*width+i;
        image[3*ind+0] = 255.0f * glm::simplex(glm::vec2(i,j+100));
        image[3*ind+1] = 0.0f; 
        image[3*ind+2] = 0.0f;

        }
    }
        
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGB32,width,height,0,GL_RGB,
    GL_FLOAT,&image[0]);

    glTexParameteri(GL_TEXTURE,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE,GL_TEXTURE_MIN_FILTER,GL_NEAREST);

    shader.use();

    while (!glfwWindowShouldClose(window)) {
  
        float current_frame = glfwGetTime(); 
        dt = current_frame - last_frame;
        last_frame = current_frame;

        processInput(window);

        glBindFramebuffer(GL_FRAMEBUFFER,fb);
        glViewport(0,0,width,height);
        glClear(GL_COLOR_BUFFER_BIT);

        shader.use();

        glm::vec2 resolution = glm::vec2(width,height);
        shader.setVec2("resolution",1,resolution);

        shader.setFloat("time",last_frame);   

        shader.setInt("seed",seed);

        shader.setBool("key_w",key_w);
        shader.setBool("key_s",key_s);
        shader.setBool("key_d",key_d);
        shader.setBool("key_a",key_a);
        shader.setBool("key_space",key_space);

        shader.setVec2("mouse",1,mouse);
    
        shader.setFloat("mouse_scroll",mouse_scroll);
        shader.setBool("mouse_pressed",mouse_pressed);
        
        glBindFramebuffer(GL_FRAMEBUFFER,0); 
        glViewport(0,0,width,height);
        glClear(GL_COLOR_BUFFER_BIT); 

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

    if(glfwGetKey(window,GLFW_KEY_F) == GLFW_PRESS)
        gli::save(tex,frag.c_str()); 

    if(glfwGetKey(window,GLFW_KEY_SPACE) == GLFW_PRESS) { 
        key_space = true;
    } else { 
        key_space = false;
    }
  
    if(glfwGetKey(window,GLFW_KEY_W) == GLFW_PRESS) {
        key_w = true; 
    } else {
        key_w = false; 
    }

    if(glfwGetKey(window,GLFW_KEY_A) == GLFW_PRESS) {
        key_a = true;
    } else {
        key_a = false;
    }

    if(glfwGetKey(window,GLFW_KEY_S) == GLFW_PRESS) { 
        key_s = true;
    } else { 
        key_s = false;
    }

    if(glfwGetKey(window,GLFW_KEY_D) == GLFW_PRESS) {
        key_d = true;
    } else {
        key_d = false;
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

