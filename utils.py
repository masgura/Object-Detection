import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import shutil
import os

class Camera:

    """" Utility class for accessing camera parameters. """

    fx = 0.0176  # focal length[m]
    fy = 0.0176  # focal length[m]
    nu = 1920  # number of horizontal[pixels]
    nv = 1200  # number of vertical[pixels]
    ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
    ppy = ppx  # vertical pixel pitch[m / pixel]
    fpx = fx / ppx  # horizontal focal length[pixels]
    fpy = fy / ppy  # vertical focal length[pixels]
    k = [[fpx,   0, nu / 2],
         [0,   fpy, nv / 2],
         [0,     0,      1]]
    K = np.array(k)

class Tango:
     
    """ Utility class for Tango body. """

    antLength = 0.3
    antZ = 0.255
    B1 =  [-0.37, 0.30,  0]
    B2 =  [-0.37, -0.26, 0]
    B3 =  [0.37,  -0.26, 0]
    B4 =  [0.37,  0.30,  0]

    S1 =  [-0.37, 0.38,  0.32]
    S2 =  [-0.37, -0.38, 0.32]
    S3 =  [0.37,  -0.38, 0.32]
    S4 =  [0.37,  0.38,  0.32]

    Ac1 = [B1[0]+0.14, B1[1], antZ]
    Ac3 = [B4[0]-0.14, B4[1], antZ]

    A1 = [-0.54, 0.49, antZ]
    A2 = [0.31, -0.56, antZ]
    A3 = [0.54, 0.49, antZ]
    Ac2 = [A2[0], B3[1], antZ]

    M = np.array([B1,B2,B3,B4,S1,S2,S3,S4,A1,A2,A3])
    M = np.concatenate((M, np.ones((len(M), 1))), axis=1) # matrix for bounding box definition, only the landmarks are defined

    T1 =  [B1[0], B1[1], 0.305]
    T2 =  [B2[0], B2[1], 0.305]
    T3 =  [B3[0], B3[1], 0.305]
    T4 =  [B4[0], B4[1], 0.305]

    M_T = np.array([T1,T2,T3,T4])
    M_T = np.concatenate((M_T, np.ones((len(M_T), 1))), axis=1)

    Ac = np.array([Ac1, Ac2, Ac3])
    Ac = np.concatenate((Ac, np.ones((len(Ac), 1))), axis=1)

def move_image(js, old_path, new_path):

    ''' Move images to a new path '''

    for i, _ in enumerate(js):
        old_image_path = old_path + js[i]["filename"]
        if os.path.exists(old_image_path): # Check that the image is in the old path
            new_image_path = new_path + js[i]["filename"]
            shutil.move(old_image_path, new_image_path)

def quat2dcm(q):

    """ Computing direction cosine matrix from quaternion """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm


def project(M, q, r):

        """ Projecting points to image frame """

        # reference points in satellite frame
        p_axes = M
        points_body = np.transpose(p_axes)

        # transformation to camera frame
        pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
        p_cam = np.dot(pose_mat, points_body)

        # getting homogeneous coordinates
        points_camera_frame = p_cam / p_cam[2]

        # projection to image plane
        points_image_plane = Camera.K.dot(points_camera_frame)

        x, y = (points_image_plane[0], points_image_plane[1])
        return x, y


def bounding_box(x, y):

    ''' Compute values of bounding box '''

    xmax = np.min((x.max(), Camera.nu))
    xmin = np.max((x.min(), 0))
    ymax = np.min((y.max(), Camera.nv))
    ymin = np.max((y.min(), 0))

    width = xmax - xmin 
    height = ymax - ymin 

    bb_x = xmax - width/2
    bb_y = ymax - height/2

    correction = np.mean([width, height])*0.1
    width += correction
    height += correction

    if width > Camera.nu:
        width = Camera.nu
    if height > Camera.nv:
        height = Camera.nv

    return bb_x, bb_y, width, height


def plot_projection(q, r, image_path):

    ''' Plot Tango wireframe onto the image plane '''

    x, y = project(Tango.M, q, r) # Main landmarks
    x_T, y_T = project(Tango.M_T, q, r) # Top main body points
    x_Ac, y_Ac = project(Tango.Ac, q, r) # Antenna clamps

    
    xb = np.append(x[0:4], x[0])
    yb = np.append(y[0:4], y[0])

    xs = np.append(x[4:8], x[4])
    ys = np.append(y[4:8], y[4])

    xt = np.append(x_T, x_T[0])
    yt = np.append(y_T, y_T[0])

    bb_x, bb_y, width, height = bounding_box(x, y)
    fig, ax = plt.subplots()

    # Load and display the background image
    image = cv2.imread(image_path)
    ax.imshow(image[:,:,::-1], extent=[0, Camera.nu, Camera.nv, 0])  

    # Plot the projected points on top of the background image
    ax.plot(xb, yb, 'b')
    ax.plot(xs, ys, 'b')
    ax.plot(xt, yt, 'b')
    ax.plot([xb[0], xt[0]], [yb[0], yt[0]], 'b')
    ax.plot([xb[1], xt[1]], [yb[1], yt[1]], 'b')
    ax.plot([xb[2], xt[2]], [yb[2], yt[2]], 'b')
    ax.plot([xb[3], xt[3]], [yb[3], yt[3]], 'b')
    ax.plot([x[8], x_Ac[0]], [y[8], y_Ac[0]], 'b')
    ax.plot([x[9], x_Ac[1]], [y[9], y_Ac[1]], 'b')
    ax.plot([x[10], x_Ac[2]], [y[10], y_Ac[2]], 'b')
    ax.plot(x, y, 'bo')


    # Create a Rectangle patch
    rect = patches.Rectangle((bb_x - width/2, bb_y - height/2), width, height, linewidth=1, edgecolor='b', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
    # Customize plot if needed
    ax.set_aspect('equal')
    ax.set_xlabel('$P_x$ [pixels]')
    ax.set_ylabel('$P_y$ [pixels]')
    plt.show()
    return

def save_labels(x, y, file_name):
    bb_x, bb_y, width, height = bounding_box(x, y)
    f = open(file_name, 'w')
    f.write('0 ' + str(bb_x/Camera.nu) + ' ' + str(bb_y/Camera.nv) + ' ' + str(width/Camera.nu) + ' ' + str(height/Camera.nv))

