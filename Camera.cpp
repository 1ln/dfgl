#include "Camera.h"

Camera::Camera(

     glm::vec3 pos,
     glm::vec3 up,
     float yawd,
     float pitchd):
     front(glm::vec3(0.0f,0.0f,-1.0f)),
     move_speed(SPEED),
     mouse_sensitivity(SENSITIVITY),
     zoom(ZOOM) {

     position = pos;
     world_up = up;
     yaw = yawd;          
     pitch = pitchd;
     
     updateVectors();
}

mat4 Camera::getViewMatrix() { 
    return glm::lookAt(position,position+front,up);
}

void Camera::processKeyboard(Cam_Move dir,float dt) {

    float velocity = move_speed * dt;
    if(dir == FORWARD)
       position += front * velocity;
    if(dir == BACKWARD)
       position += front * velocity;
    if(dir == LEFT)
       position += right * velocity;
    if(dir == RIGHT)
       position += right * velocity;
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

updateVectors();
}

void Camera::processMouseScroll(float yoff) {

    zoom -= (float)yoff;

    if(zoom < 1.0f)
        zoom = 1.0f;
    if(zoom > 45.0f)
        zoom = 45.0f;
}

void Camera::updateVectors() {
   
    glm::vec3 nfront;

    nfront.x = cos(glm:radians(yaw)) * cos(glm::radians(pitch));
    nfront.y = sin(glm::radians(pitch)); 
    nfront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

    front = glm::normalize(nfront);

    right = glm::normalize(glm::cross(front,world_up));
    up = glm::normalize(glm::cross(right,front));
 
}
