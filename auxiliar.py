import numpy as np
import cv2

def depth_on_floor(camera, u, v):
    """
    Compute the camera‐frame depth Z_c of the ray through pixel (u,v)
    that hits the world‐floor Z_w=0.
    """
    # 1) Get intrinsics & extrinsics
    K, W, H      = camera.getIntrinsics()
    R_wc, t_wc   = camera.getExtrinsics()   # world→camera
    R_c2w        = R_wc.T                   # camera→world
    C_w          = -R_c2w @ t_wc            # camera center in world coords (3×1)

    # 2) Ray direction in camera frame
    K_inv        = np.linalg.inv(K)
    ray_cam      = (K_inv @ np.array([u, v, 1.0]).reshape(3,1)).flatten()  # 3-vector

    # 3) Express ray in world coords:
    #    X_w(s) = C_w + s*(R_c2w @ ray_cam)
    ray_w        = (R_c2w @ ray_cam.reshape(3,1)).flatten()  # 3-vector direction
    Cz           = C_w[2,0]                                  # camera’s world-Z
    dz           = ray_w[2]                                  # direction’s world-Z

    if abs(dz) < 1e-6:
        return None  # parallel to floor, no intersection

    # solve C_w.z + s*dz = 0  →  s = -Cw_z / dz
    s = -Cz / dz

    # now depth in camera frame is s * (ray_cam’s Z component)
    depth = s * ray_cam[2]

    return float(depth)

def segment_yellow_borders(img_rgb):
    # 1) Convert to HSV
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # 2) Define yellow range (tune these!)
    lower = np.array([20, 100, 100])
    upper = np.array([40, 255, 255])

    # 3) Threshold
    mask = cv2.inRange(hsv, lower, upper)

    # 4) (Optional) Morphological closing to fill gaps
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 5) (Optional) Erosion to remove noise
    mask = cv2.erode(mask, kernel, iterations=1)

    # Inverting Colors
    #mask = cv2.bitwise_not(mask)

    return mask

def extract_wall_lines(mask, 
                        rho=1, 
                        theta=np.pi/180, 
                        threshold=10, 
                        min_line_length=10, 
                        max_line_gap=10):
    """
    Detects dominant straight borders in a binary mask using Probabilistic Hough Transform.
    Returns a list of line segments, each as ((x1, y1), (x2, y2)).
    """
    # 1) Edge detection
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)

    # 2) Hough line detection (Probabilistic)
    raw_lines = cv2.HoughLinesP(edges,
                                rho=rho,
                                theta=theta,
                                threshold=threshold,
                                minLineLength=min_line_length,
                                maxLineGap=max_line_gap)
    if raw_lines is None:
        return []

    # 3) Collect and optionally merge near‐collinear segments
    lines = [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in raw_lines[:,0]]

    # 4) (Optional) Merge lines with similar angle and close distance
    merged = []
    for line in lines:
        added = False
        (x1, y1), (x2, y2) = line
        angle = np.arctan2(y2 - y1, x2 - x1)
        for m in merged:
            (mx1, my1), (mx2, my2), mangle = m
            if abs(np.sin(angle - mangle)) < 0.1:
                # endpoints close?
                if (np.hypot(x1-mx1, y1-my1) < 20 or 
                    np.hypot(x1-mx2, y1-my2) < 20 or
                    np.hypot(x2-mx1, y2-my1) < 20 or
                    np.hypot(x2-mx2, y2-my2) < 20):
                    # merge by extending endpoints
                    pts = np.array([[x1,y1],[x2,y2],[mx1,my1],[mx2,my2]])
                    dists = pts.dot([np.cos(angle), np.sin(angle)])
                    idx_min, idx_max = np.argmin(dists), np.argmax(dists)
                    new_p1 = tuple(pts[idx_min])
                    new_p2 = tuple(pts[idx_max])
                    m[:] = [new_p1, new_p2, angle]
                    added = True
                    break
        if not added:
            merged.append([ (x1,y1), (x2,y2), angle ])

    # strip angles
    wall_lines = [ (l[0], l[1]) for l in merged ]
    return wall_lines
