import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import cv2

# 1) Conexión
client = RemoteAPIClient()
sim = client.getObject('sim')

# 2) Handles
camera_handle = sim.getObject('/SkyCamera')
robot_handle  = sim.getObject('/PioneerP3DX')

# 3) Funciones auxiliares (euler→R, extrínsecos e intrínsecos)
def euler_to_rot_matrix(euler):
    alpha, beta, gamma = euler
    Rz = np.array([
        [ np.cos(alpha), -np.sin(alpha), 0 ],
        [ np.sin(alpha),  np.cos(alpha), 0 ],
        [             0,              0, 1 ]
    ])
    Ry = np.array([
        [  np.cos(beta), 0, np.sin(beta) ],
        [             0, 1,            0 ],
        [ -np.sin(beta), 0, np.cos(beta) ]
    ])
    Rx = np.array([
        [ 1,            0,             0 ],
        [ 0, np.cos(gamma), -np.sin(gamma) ],
        [ 0, np.sin(gamma),  np.cos(gamma) ]
    ])
    return Rz.dot(Ry).dot(Rx)

def get_camera_extrinsics():
    """
    Devuelve (R_wc, t_wc), la rotación y traslación mundo→cámara.
    """
    # 1) Leer posición de la cámara en el mundo
    cam_pos = sim.getObjectPosition(camera_handle, -1)  # [x_c, y_c, z_c]
    t_c = np.array([[cam_pos[0]], [cam_pos[1]], [cam_pos[2]]])  # 3×1

    # 2) Leer orientación en Euler ZYX
    cam_euler = sim.getObjectOrientation(camera_handle, -1)    # [alpha, beta, gamma]
    R_c = euler_to_rot_matrix(cam_euler)                       # Rotación cámara→mundo

    # 3) Inversa: mundo→cámara
    R_wc = R_c.T
    t_wc = - R_c.T.dot(t_c)

    return R_wc, t_wc

def get_camera_intrinsics():
    """
    Consulta la SkyCamera para obtener:
      - Resolución (W, H)
      - FOV vertical en radianes (fov_y), usando sim.getObjectFloatParameter
    A partir de eso calcula:
      f_y = (H/2) / tan(fov_y/2)
      f_x = f_y * (W/H)
      c_x = W/2
      c_y = H/2

    Devuelve (K, W, H), donde K es la matriz intrínseca 3×3:
        [ f_x   0    c_x ]
        [  0   f_y   c_y ]
        [  0    0     1  ]
    """
    # 1) Leer resolución de la SkyCamera
    res = sim.getVisionSensorResolution(camera_handle)  # devuelve [W, H]
    W, H = res[0], res[1]

    # 2) Leer FOV vertical (radianes) de la SkyCamera
    _, fov_y = sim.getObjectFloatParameter(
        camera_handle,
        sim.visionfloatparam_perspective_angle
    )

    # 3) Calcular f_y, f_x, c_x, c_y
    f_y = (H / 2.0) / np.tan(fov_y / 2.0)
    f_x = f_y * (W / float(H))
    c_x = W / 2.0
    c_y = H / 2.0

    K = np.array([
        [ f_x,  0.0, c_x ],
        [ 0.0,  f_y, c_y ],
        [ 0.0,  0.0,  1.0 ]
    ])

    return K, W, H

def world_to_pixel(X_w, Y_w, Z_w=0.0):
    """
    Proyecta (X_w, Y_w, Z_w) → (u, v), corrigiendo el espejado en X.
    Retorna None si Z_c ≤ 0 (punto detrás de la cámara).
    """
    # 1) Coordenadas mundo
    p_w = np.array([[X_w], [Y_w], [Z_w]])

    # 2) Extrínsecos (mundo→cámara)
    R_wc, t_wc = get_camera_extrinsics()

    # 3) Convertir a cámara: p_c = R_wc·p_w + t_wc
    p_c = R_wc.dot(p_w) + t_wc
    X_c, Y_c, Z_c = p_c[0,0], p_c[1,0], p_c[2,0]
    if Z_c <= 0:
        return None

    # 4) Normalizar (invirtiendo X_c para corregir el espejado en la imagen)
    x_n = -X_c / Z_c
    y_n = -Y_c / Z_c - 0.12

    # 5) Intrínsecos
    K, W, H = get_camera_intrinsics()
    f_x = K[0,0]
    f_y = K[1,1]
    c_x = K[0,2]
    c_y = K[1,2]

    # 6) Proyección
    u = f_x * x_n + c_x
    v = f_y * y_n + c_y
    return (u, v)


# ------------------------------------------------------
# 4) Loop de ejemplo: leer imagen + proyectar la posición
# ------------------------------------------------------
if __name__ == "__main__":
    # 4.1) Iniciar simulación en modo “paso a paso” (stepping)
    client.setStepping(True)
    sim.startSimulation()

    time.sleep(0.5)  # esperar medio segundo antes de la primera lectura

    # 4.2) Bucle: por ejemplo 100 iteraciones
    for _ in range(100):
        # (a) Tomar la imagen con sim.getVisionSensorCharImage → (buffer, W, H)
        img_buffer, resX, resY = sim.getVisionSensorCharImage(camera_handle)
        # Convertir buffer a array NumPy dtype=uint8
        arr = np.frombuffer(img_buffer, dtype=np.uint8).reshape((resY, resX, 3))
        # CoppeliaSim devuelve (x de izquierda a derecha, y de abajo hacia arriba, RGB).
        # Para mostrar en OpenCV: BGR y voltear verticalmente:
        img_bgr = cv2.flip(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), 0)

        # (b) Guardar un duplicado “crudo” de la imagen antes de dibujar
        img_raw = img_bgr.copy()

        # (c) Leer posición mundial del robot
        pos_robot = sim.getObjectPosition(robot_handle, -1)  # [Xw, Yw, Zw]
        Xw, Yw, Zw = pos_robot[0], pos_robot[1], pos_robot[2]

        # (d) Proyectar a píxel (u, v) sobre la copia “overlay”
        img_overlay = img_bgr  # apuntamos al mismo array de img_bgr para dibujar encima
        uv = world_to_pixel(Xw, Yw, Zw)
        if uv is not None:
            u, v = uv
            u_int = int(round(u))
            v_int = int(round(v))
            if 0 <= u_int < resX and 0 <= v_int < resY:
                cv2.circle(img_overlay, (u_int, v_int), 5, (0, 0, 255), -1)

        # (e) Mostrar las dos ventanas simultáneamente
        cv2.imshow("SkyCamera (Raw Frame)", img_raw)
        cv2.imshow("SkyCamera (With Projection)", img_overlay)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        # (f) Avanzar un paso de simulación
        client.step()

    # 4.3) Detener simulación y cerrar ventanas
    sim.stopSimulation()
    cv2.destroyAllWindows()
