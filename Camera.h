#pragma once

enun Cam_Move {
  
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT

};

class Camera {

public:

    Camera(glm::vec3 pos,
           glm::vec3 tar,
           glm::vec3 up,
           glm::vec3 front,
           glm::vec3 right);
    
    void setPos(glm::vec3 pos);
    void setTar(glm::vec3 tar);
     
    glm::mat4 getViewMatrix();

    void processKeyboard(Cam_Move dir,float dt);
    void processMouseMove(float xoff,float yoff,bool constraint = true);
    void processMouseScroll(float yoff);
    
private:

    void updateVectors();

    glm::vec3 pos_;
    glm::vec3 front_;
    glm::vec3 right_;
    glm::vec3 up_;

    float yaw_;
    float pitch_;

    float fov_;

    float moveSpeed_;
    float mouseSensitivity_;

         

