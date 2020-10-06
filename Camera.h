#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>

enum Cam_Move {
  
    SPACE,
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT,
    YAW_LEFT,
    PITCH_FORWARD,
    YAW_RIGHT,
    PITCH_BACKWARD,

};

const float YAW = -1.57f;
const float PITCH = 0.0f;
const float SPEED = 2.25f;
const float SENSITIVITY = 0.1f;
const float ZOOM = 45.0f;

class Camera {

public:

    Camera(glm::vec3 pos = glm::vec3(0.0f,0.0f,0.0f),
           glm::vec3 tar = glm::vec3(0.0f,0.0f,0.0f), 
           glm::vec3 upd  = glm::vec3(0.0f,1.0f,0.0f),
           float yawd    = YAW,
           float pitchd  = PITCH);
                    
    glm::mat4 getViewMatrix();

    void update();

    void processKeyboard(Cam_Move dir,float dt);
    void processMouseMove(float xoff,float yoff,bool limit_pitch);
    void processMouseScroll(float yoff);
    
    glm::vec3 position;
    glm::vec3 target;    
    glm::vec3 front;
    glm::vec3 right;
    glm::vec3 up;

    float yaw;
    float pitch;

    float zoom;

    float move_speed;
    float mouse_sensitivity;


};
