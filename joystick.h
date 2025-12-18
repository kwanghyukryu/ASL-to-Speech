#ifndef HAL_JOYSTICK_H
#define HAL_JOYSTICK_H

#include <stdbool.h>

typedef enum {
    JOYSTICK_NONE,
    JOYSTICK_UP,
    JOYSTICK_DOWN,
    JOYSTICK_LEFT,
    JOYSTICK_RIGHT
} JoystickDirection;


void Joystick_init(void);


JoystickDirection Joystick_get_direction(void);

bool Joystick_is_pressed(void);


void Joystick_cleanup(void);

#endif 