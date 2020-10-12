#include "Camera.h"

Camera::Camera(

     glm::vec3 pos,
     glm::vec3 tar,
     glm::vec3 upd,
     float yawd,
     float pitchd):
     front(glm::vec3(0.0f,0.0f,-1.0f)),
     move_speed(SPEED),
     mouse_sensitivity(SENSITIVITY),
     zoom(ZOOM) {

     position = pos;
     wup = upd;
     yaw = yawd;          
     pitch = pitchd;
     
}

glm::mat4 Camera::getViewMatrix() { 
    return glm::lookAt(position,position+front,up);
}

void Camera::processKeyboard(Cam_Move dir,float dt) {

    float velocity = move_speed * dt;

    if(dir == FORWARD)
       position += front * velocity;

    if(dir == BACKWARD)
       position -= front * velocity;

    if(dir == LEFT)
       position += right * velocity;

    if(dir == RIGHT)
       position -= right  * velocity;

    if(dir == UP)
       position += up * velocity;

    if(dir == PITCH_FORWARD)  
       front += glm::normalize(front * cosf(pitch) -      
                               up    * sinf(pitch) ) *   
                               velocity;
 
    if(dir == PITCH_BACKWARD)    
       front += glm::normalize(front * cosf(pitch) +       
                               up    * sinf(pitch) ) *
                               velocity;

    if(dir == YAW_LEFT) 
       front += glm::normalize(front * cosf(yaw) -
                               right * sinf(yaw) ) *
                               velocity;

    if(dir == YAW_RIGHT) 
       front += glm::normalize(front * cosf(yaw) +  
                               right * sinf(yaw) ) * 
                               velocity;
}

void Camera::processMouseMove(float xoff,float yoff,bool limit_pitch = true) {
xoff *= mouse_sensitivity;
yoff *= mouse_sensitivity; 

yaw += xoff;
pitch += yoff;

if(limit_pitch) {
    if(pitch > 89.0f)
        pitch = 89.0f;
    if(pitch < -89.0f)
        pitch = -89.0f; 
}
}

void Camera::processMouseScroll(float yoff) {

    zoom -= (float)yoff;

    if(zoom < 1.0f)
        zoom = 1.0f;
    if(zoom > 45.0f)
        zoom = 45.0f;
}

void Camera::update() {
   
    right = glm::normalize(glm::cross(front,wup));
    up = glm::normalize(glm::cross(right,front));

    target = position + front;
 
}
