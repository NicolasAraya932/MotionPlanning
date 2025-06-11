import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import cv2

def euler_to_rot_matrix(alpha, beta, gamma):
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

def pixel_to_world_coordinates(camera, pixel_coords, depth):
    """
    Convierte coordenadas de píxel (u,v) a coordenadas del mundo (X_w,Y_w,Z_w),
    dado:
      K       : 3×3 intrinsics
      R_wc    : 3×3 rotación world->camera
      t_wc    : 3×1 traslación world->camera
      pixel_coords: (u, v)
      depth   : Z_c, la profundidad real del punto en la cámara (en las mismas unidades que t_wc)

    Devuelve:
      world_coords: (X_w, Y_w, Z_w)
    """
    u, v = pixel_coords

    K,W,H = camera.getIntrinsics()  # K: 3×3, W: ancho, H: alto
    # 1) Ray direction in camera frame (normalized by K):
    K_inv = np.linalg.inv(K)
    ray_camera = K_inv @ np.array([u, v, 1.0])  # 3×1

    # 2) Scale by actual depth to get the 3D point in camera frame:
    X_c = ray_camera * depth                     # 3×1

    R_wc, t_wc = camera.getExtrinsics()  # R_wc: 3×3, t_wc: 3×1

    # 3) Transform back to world frame:
    #    X_w = R_wc.T @ (X_c - t_wc)
    #    (note: t_wc is world->camera, so X_c = R_wc X_w + t_wc → invert)
    world_coords = R_wc.T @ (X_c.reshape(3,1) - t_wc)

    return world_coords.flatten()

def world_to_pixel_coordinates(camera, X_w, Y_w, Z_w=0.0):
    """
    Proyecta (X_w, Y_w, Z_w) → (u, v).
    Retorna None si Z_c ≤ 0 (punto detrás de la cámara).
    """
    # 1) Punto en mundo como vector 3×1
    p_w = np.array([[X_w], [Y_w], [Z_w]])  # 3×1

    # 2) Obtener extrínsecos world→camera
    R_wc, t_wc = camera.getExtrinsics()     # R_wc: 3×3, t_wc: 3×1

    # 3) Transformar al sistema de la cámara: X_c = R_wc * X_w + t_wc
    p_c = R_wc @ p_w + t_wc                # 3×1
    X_c, Y_c, Z_c = p_c.flatten()
    if Z_c <= 0:
        return None

    # 4) Construir matriz 3×4 [R|t]
    Rt = np.hstack((R_wc, t_wc))           # 3×4

    # 5) Obtener intrínsecos completos (incluye skew γ si existe)
    K, W, H = camera.getIntrinsics()       # K: 3×3

    # 6) Proyección homogénea: x_hom = K @ [R|t] @ [X_w, Y_w, Z_w, 1]^T
    Xw_hom = np.array([X_w, Y_w, Z_w, 1.0]).reshape(4,1)  # 4×1
    x_hom = K @ (Rt @ Xw_hom)                             # 3×1

    # 7) Normalización con s = x_hom[2]
    s = x_hom[2,0]
    u = x_hom[0,0] / s
    v = x_hom[1,0] / s

    return (u, v)

class CameraCoppelia:

    def __init__(self, sim, camera_name='/SkyCamera'):

        self.sim = sim
        self.camera_name = camera_name
        self.camera_handle = sim.getObject(camera_name)
        self.position = self.sim.getObjectPosition(self.camera_handle, -1)  # [x_c, y_c, z_c]
        self.orientation = self.sim.getObjectOrientation(self.camera_handle, -1)  # [alpha, beta, gamma]

    
    def getExtrinsics(self):
        """
        Devuelve (R_wc, t_wc), la rotación y traslación mundo→cámara.
        """
        # 1) Leer posición de la cámara en el mundo
        cam_pos = self.sim.getObjectPosition(self.camera_handle, -1)  # [x_c, y_c, z_c]
        t_c = np.array([[cam_pos[0]], [cam_pos[1]], [cam_pos[2]]])  # 3×1

        # 2) Leer orientación en Euler ZYX
        alpha, beta, gamma = self.sim.getObjectOrientation(self.camera_handle, -1)    # [alpha, beta, gamma]
        R_c = euler_to_rot_matrix(alpha, beta, gamma)                       # Rotación cámara→mundo

        # 3) Inversa: mundo→cámara
        R_wc = R_c.T
        t_wc = - R_c.T.dot(t_c)

        return R_wc, t_wc

    def getIntrinsics(self):

        # 1) Leer resolución de la SkyCamera
        res = self.sim.getVisionSensorResolution(self.camera_handle)  # devuelve [W, H]
        W, H = res[0], res[1]

        # 2) Leer FOV vertical (radianes) de la SkyCamera
        _, fov_y = self.sim.getObjectFloatParameter(
            self.camera_handle,
            self.sim.visionfloatparam_perspective_angle
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
    
    def getFrame(self):
        
        img_buffer, resX, resY = self.sim.getVisionSensorCharImage(self.camera_handle)
        # Convertir buffer a array NumPy dtype=uint8
        arr = np.frombuffer(img_buffer, dtype=np.uint8).reshape((resY, resX, 3))
        # CoppeliaSim devuelve (x de izquierda a derecha, y de abajo hacia arriba, RGB).
        # Para mostrar en OpenCV: BGR y voltear verticalmente:
        #img_bgr = cv2.flip(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), 0)

        # (b) Guardar un duplicado “crudo” de la imagen antes de dibujar
        return arr, resX, resY