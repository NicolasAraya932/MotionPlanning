import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import cv2
from cameraCoppelia import CameraCoppelia, world_to_pixel_coordinates, pixel_to_world_coordinates
import matplotlib.pyplot as plt
from auxiliar import *
from astar import astar
from PID import *

client = RemoteAPIClient()
client.setStepping(True)

sim = client.getObject('sim')
# 2) Handles
robot_handle  = sim.getObject('/PioneerP3DX')
robot_gps_handle = sim.getObject('/PioneerP3DX/GPS')
goal_gps_handle = sim.getObject('/goal/gps')

camera = CameraCoppelia(sim, '/SkyCamera')

sim.startSimulation()
for _ in range(1):
    client.step()
    time.sleep(0.5)
    client.step()
    cv2_frame, resx, resy = camera.getFrame()

    # 1. Get Robot and Goal Positions (in world coordinates)
    robot_pos = sim.getObjectPosition(robot_gps_handle, -1)
    print('Robot Position:', robot_pos)
    robot_u, robot_v = world_to_pixel_coordinates(camera, robot_pos[0], robot_pos[1], robot_pos[2])

    goal_pos = sim.getObjectPosition(goal_gps_handle, -1)
    print('Goal Position:', goal_pos)
    goal_u, goal_v = world_to_pixel_coordinates(camera, goal_pos[0], goal_pos[1], goal_pos[2])  # pixel coordinates

    # 2. Segment walls from image
    mask = segment_yellow_borders(cv2_frame)

    # 3. Extract walls using Hough transform
    walls_uv = extract_wall_lines(mask)

    # 4. Create occupancy grid (1 = free, 0 = obstacle)
    grid = np.ones_like(mask, dtype=np.uint8)  # shape=(H,W)
    for (u1, v1), (u2, v2) in walls_uv:
        cv2.line(grid, (u1, v1), (u2, v2), 0, thickness=3)

    # Optional: make walls thicker to account for robot size
    kernel = np.ones((3,3), np.uint8)
    grid = cv2.erode(grid, kernel)

    # Convertir dimensiones del robot a píxeles
    K, W, H = camera.getIntrinsics()
    fx = K[0, 0]
    fy = K[1, 1]

    # Robot ocupa una elipse/círculo de radio aproximadamente (en píxeles)
    robot_radius_x_px = int((0.2 / 2.0) * fx)
    robot_radius_y_px = int((0.2 / 2.0) * fy)

    # Crear el kernel elíptico (estructura del robot)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * robot_radius_x_px, 2 * robot_radius_y_px))

    # Inflar los obstáculos (0s) para que representen el espacio de colisión
    inflated_grid = cv2.dilate(1 - grid, kernel)  # Invertir para que los obstáculos sean 1s → dilatar → revertir
    inflated_grid = 1 - inflated_grid  # Volver a formato 1=libre, 0=obstáculo

    # A*
    start = (int(robot_v), int(robot_u))  # asegúrate de usar coordenadas y,x
    goal  = (int(goal_v), int(goal_u))
    path = astar(inflated_grid, start, goal)

    if path:
        path = path[::50]  # reduce path resolution
    
    # Remove the first point if it's the robot's position
    if path and path[0] == start:
        path = path[1:]

    if path:
        # dibujar puntos con opencv
        for (y, x) in path:
            cv2.circle(cv2_frame, (x, y), 3, (255, 0, 0), -1)

    else:
        print("No path found.")

sim.stopSimulation()

time.sleep(0.5)

path_world = [pixel_to_world_coordinates(camera, (x, y), depth_on_floor(camera, x, y)) for (y, x) in path]

# PID controller for heading
pid = PID(Kp=2.5, Ki=0.0, Kd=0.4)

left_motor = sim.getObject('/PioneerP3DX/leftMotor')
right_motor = sim.getObject('/PioneerP3DX/rightMotor')

# Tuning params
linear_speed = 0.7  # m/s
goal_tolerance = 0.2  # m
dt = 0.05  # control step

sim.startSimulation()
time.sleep(0.5)

for _ in range(1000):
    client.step()
    time.sleep(dt)

    cv2_frame, resx, resy = camera.getFrame()

    # Get robot pose
    robot_pos = sim.getObjectPosition(robot_handle, -1)
    robot_ori = sim.getObjectOrientation(robot_handle, -1)
    robot_theta = robot_ori[2]  # yaw

    # Display robot position as cv2.circle 
    cv2.circle(cv2_frame, (int(robot_pos[0] * resx), int(robot_pos[1] * resy)), 5, (0, 255, 0), -1)

    # Find nearest point in path_world
    dists = [distance_2d(robot_pos, p) for p in path_world]
    nearest_idx = int(np.argmin(dists))

    # Stop if close to goal
    if nearest_idx >= len(path_world) - 2:
        print("Goal reached")
        sim.setJointTargetVelocity(left_motor, 0)
        sim.setJointTargetVelocity(right_motor, 0)
        break

    target = path_world[nearest_idx + 1]
    dx = target[0] - robot_pos[0]
    dy = target[1] - robot_pos[1]

    # Compute desired heading
    desired_theta = atan2(dy, dx)
    heading_error = normalize_angle(desired_theta - robot_theta)

    # PID for angular velocity
    omega = pid.update(heading_error, dt)

    # Constant forward speed if aligned
    distance = distance_2d(robot_pos, target)
    v = linear_speed if abs(heading_error) < pi / 4 else 0.0

    # Convert to wheel velocities
    left_motor_rel = sim.getObjectPosition(left_motor, robot_handle)
    right_motor_rel = sim.getObjectPosition(right_motor, robot_handle)

    L = np.linalg.norm(np.array(left_motor_rel[:2]) - np.array(right_motor_rel[:2]))
    vl = v - omega * L / 2
    vr = v + omega * L / 2

    sim.setJointTargetVelocity(left_motor, vl)
    sim.setJointTargetVelocity(right_motor, vr)

    cv2.imshow("",cv2_frame)

sim.stopSimulation()