#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp> 

#include "Shader.h"
#include "Camera.h"

#include <iostream>
#include <vector>
#include <string>

void framebuffer_size_callback(GLFWwindow* window,int w,int h);
void mouse_callback(GLFWwindow* window,double xpos,double ypos);
void mouse_button_callback(GLFWwindow* window,int button,int action,int mods); 
void scroll_callback(GLFWwindow* window,double xoff,double yoff); 
void key_callback(GLFWwindow* window,int key,int scancode,int action,int mods);  
void processInput(GLFWwindow *window);
void saveImage(char* filepath,GLFWwindow* window);

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

bool key_up = false;
bool key_down = false;
bool key_right = false; 
bool key_left = false;
bool key_x = false;
bool key_z = false;

bool hide_cursor = false;

Camera cam(glm::vec3(0.0,0.0,5.0),
           glm::vec3(0.0,0.0,0.0)); 

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

    Shader shader("vert.glsl",frag.c_str());         
   
    shader.use();
    
    while (!glfwWindowShouldClose(window)) {
  
        float current_frame = glfwGetTime(); 
        dt = current_frame - last_frame;
        last_frame = current_frame;

        processInput(window);
        glfwSetKeyCallback(window,key_callback);

        glm::vec2 resolution = glm::vec2(width,height);        

        glBindFramebuffer(GL_FRAMEBUFFER,fb);
        shader.use();
        shader.setVec2("resolution",1,resolution);
        shader.setFloat("time",last_frame);
        shader.setVec3("camPos",1,cam.position);
        shader.setVec2("mouse",1,mouse);
        shader.setFloat("mouse_scroll",mouse_scroll);
        shader.setBool("mouse_pressed",mouse_pressed);        
        shader.setBool("up",key_up);
        shader.setBool("down",key_down);
        shader.setBool("right",key_right);
        shader.setBool("left",key_left);
        shader.setBool("key_x",key_x);
        shader.setBool("key_z",key_z);

        glBindFramebuffer(GL_FRAMEBUFFER,0);  

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

void key_callback(GLFWwindow* window,int key,int scancode,int action,int mods) {

    if(key == GLFW_KEY_UP && action == GLFW_PRESS)
        key_up = true;
    if(key == GLFW_KEY_UP && action == GLFW_RELEASE)
        key_up = false;

    if(key == GLFW_KEY_DOWN && action == GLFW_PRESS)
        key_down = true;
    if(key == GLFW_KEY_DOWN && action == GLFW_RELEASE)
        key_down = false;

    if(key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
        key_right = true;
    if(key == GLFW_KEY_RIGHT && action == GLFW_RELEASE)
        key_right = false;

    if(key == GLFW_KEY_LEFT && action == GLFW_PRESS)
        key_left = true;
    if(key == GLFW_KEY_LEFT && action == GLFW_RELEASE)
        key_left = false;

    if(key == GLFW_KEY_X && action == GLFW_PRESS)
        key_x = true;
    if(key == GLFW_KEY_X && action == GLFW_RELEASE)
        key_x = false;
       
    if(key == GLFW_KEY_Z && action == GLFW_PRESS)
        key_z = true;
    if(key == GLFW_KEY_Z && action == GLFW_RELEASE)
        key_z = false;

}

void saveImage(char* filepath,GLFWwindow* window) {

    int w,h;
    glfwGetFramebufferSize(window,&w,&h);
    GLsizei n = 3;
    GLsizei stride = n * w;
    stride += (stride % 4) ? (4 - stride % 4) : 0;
    GLsizei buffersize = stride * h;
    std::vector<char> buffer(buffersize);
    glPixelStorei(GL_PACK_ALIGNMENT,4);
    glReadBuffer(GL_FRONT);
    glReadPixels(0,0,w,h,GL_RGB,GL_UNSIGNED_BYTE,buffer.data());



}
