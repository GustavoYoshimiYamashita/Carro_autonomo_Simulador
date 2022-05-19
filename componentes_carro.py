def set_steering_angle(wheel_angle, left_steer, right_steer):
    steering_angle = wheel_angle
    left_steer.setPosition(steering_angle)
    right_steer.setPosition(steering_angle)

def set_speed(velocity, left_front_wheel, right_front_wheel, left_rear_wheel, right_rear_wheel):
    left_front_wheel.setVelocity(velocity)
    right_front_wheel.setVelocity(velocity)
    left_rear_wheel.setVelocity(velocity*0.7)
    right_rear_wheel.setVelocity(velocity*0.7)