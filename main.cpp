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

glm::vec3 cam_position;
glm::vec3 cam_target;
glm::vec3 fov;

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

    unsigned int tex;
    glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D,tex);

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

        shader.setInt("seed",seed);

        shader.setint("aa",aa);

        shader.setVec3("cam_position",1,cam_position);        
        shader.setVec3("cam_target",1,cam_target);
        shader.setFloat("fov",fov);

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
        

    if(glfwGetKey(window,GLFW_KEY_A) == GLFW_PRESS)
       

    if(glfwGetKey(window,GLFW_KEY_S) == GLFW_PRESS)
       

    if(glfwGetKey(window,GLFW_KEY_D) == GLFW_PRESS)
        

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

    float x = (2.0f * xpos ) / width - 1.0f;
    float y = 1.0f - (2.0f * ypos ) / height;
    glm::vec3 ray_nds = glm::vec3(x,y,1);
    cam_target = ray_nds;

}

void scroll_callback(GLFWwindow* window,double xoff, double yoff) {
    fov -= float(yoff);
}

