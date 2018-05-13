import numpy as np
import cv2
import math


def depth_image_to_real(depth_image, rgb_image):
    """
    convert depth image to real depth image
    :param depth_image: depth image
    :param rgb_image: rgb image
    :return: real depth image
    """
    depth_image = np.float32(depth_image)
    depth_real = np.zeros((depth_image.shape[0], depth_image.shape[1], rgb_image.shape[2]), dtype=rgb_image.dtype)

    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):

            Zd = depth_image[i][j]
            if Zd != 0:
                m1 = np.array((j, i, 1)) * Zd
                m2 = np.matmul(m1, inv_intrinsic_IR)
                m3 = np.matmul(m2, r) + t
                m4 = np.matmul(m3, intrinsic_RGB)

                colc = (int)((m4[0][0] / m4[0][2]))
                rowc = (int)((m4[0][1] / m4[0][2]))

                if (0 < colc < np.shape(rgb_image)[1]) and (0 < rowc < np.shape(rgb_image)[0]):
                    for ch in range(0, 3):
                        depth_real[i][j][ch] = rgb_image[rowc][colc][ch]

    return depth_real


def velocity(depth_image_1, depth_image_2, x1, y1, x2, y2):
    """
    calculate the velocity of the ball given two depth image and its center in both the images
    :param depth_image_1: starting depth image
    :param depth_image_2: final depth image
    :param x1: x coordinate of ball center in starting depth image
    :param y1: y coordinate of ball center in starting depth image
    :param x2: x coordinate of ball center in final depth image
    :param y2: y coordinate of ball center in final depth image
    :return: velocity of the ball
    """
    z1 = depth_image_1[y1][x1]
    colD1 = (y1 * inv_intrinsic_IR[0][0] + x1 * inv_intrinsic_IR[1][0] + 1 * inv_intrinsic_IR[2][0]) * z1
    rowD1 = (y1 * inv_intrinsic_IR[0][1] + x1 * inv_intrinsic_IR[1][1] + 1 * inv_intrinsic_IR[2][1]) * z1
    zowD1 = (y1 * inv_intrinsic_IR[0][2] + x1 * inv_intrinsic_IR[1][2] + 1 * inv_intrinsic_IR[2][2]) * z1

    z2 = depth_image_2[y2][x2]
    colD2 = (y2 * inv_intrinsic_IR[0][0] + x2 * inv_intrinsic_IR[1][0] + 1 * inv_intrinsic_IR[2][0]) * z2
    rowD2 = (y2 * inv_intrinsic_IR[0][1] + x2 * inv_intrinsic_IR[1][1] + 1 * inv_intrinsic_IR[2][1]) * z2
    zowD2 = (y2 * inv_intrinsic_IR[0][2] + x2 * inv_intrinsic_IR[1][2] + 1 * inv_intrinsic_IR[2][2]) * z2

    velX = (colD2 - colD1) / time_frame
    velY = (rowD2 - rowD1) / time_frame
    velZ = (zowD2 - zowD1) / time_frame
    vel = math.sqrt(math.pow(velX, 2) + math.pow(velY, 2) + math.pow(velZ, 2))
    return vel


def ball_detect(color_depth, color):
    """
    detect the ball in a given depth image for a given color
    :param color_depth: depth image
    :param color: color of the ball
    :return: x, y coordinate of ball center
    """
    hsv_image = cv2.cvtColor(color_depth, cv2.COLOR_BGR2HSV)
    if color == 'brown':
        low = np.array([12, 30, 30])
        high = np.array([20, 255, 255])
    else:
        low = np.array([1, 30, 30])
        high = np.array([12, 255, 255])

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.inRange(hsv_image, low, high)
    cv2.dilate(mask, kernel, mask, iterations=2)
    cv2.erode(mask, kernel, mask, iterations=8)
    cv2.dilate(mask, kernel, mask, iterations=8)
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    height, width, _ = np.shape(color_depth)
    min_x, min_y = width, height
    max_x = max_y = 0
    for contour, hier in zip(contours, hierarchy):
        (x, y, w, h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x + w, max_x)
        min_y, max_y = min(y, min_y), max(y + h, max_y)

    if max_x - min_x > 0 and max_y - min_y > 0:
        cv2.rectangle(color_depth, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
        x_center = (min_x + max_x) / 2
        y_center = (min_y + max_y) / 2

    return int(x_center), int(y_center)


# variable declaration
start_image = cv2.imread("image/data/color-63647317626781.png")
final_image = cv2.imread("image/data/color-63647317628081.png")
start_depth_image = cv2.imread("image/data/depth-63647317626781.png", cv2.IMREAD_ANYDEPTH)
final_depth_image = cv2.imread("image/data/depth-63647317628081.png", cv2.IMREAD_ANYDEPTH)

intrinsic_RGB = np.array(((1034.3, 0, 0), (1.6609, 1034.3, 0), (970.63, 529.07, 1)))
inv_intrinsic_IR = np.array(((0.0027132, 0, 0), (-1.8517e-05, 0.0026991, 0), (-0.69724, -0.54233, 1)))
transformation_D_C = np.array(((0.99994, -0.0090793, 0.0073288, 0), (0.009138, 0.99992, -0.0079812, 0),
                               (-0.0072615, 0.0080413, 0.99995, 0), (37.331, -42.241, -88.088, 1)))
r = transformation_D_C[0:3, 0:3]
t = transformation_D_C[3:, 0:3]

time_frame = 1.300

start_depth_real = depth_image_to_real(start_depth_image, start_image)
final_depth_real = depth_image_to_real(final_depth_image, final_image)


rxcntr1, rycntr1 = ball_detect(start_depth_real, 'red')
rxcntr2, rycntr2 = ball_detect(final_depth_real, 'red')

bxcntr1, bycntr1 = ball_detect(start_depth_real, 'brown')
bxcntr2, bycntr2 = ball_detect(final_depth_real, 'brown')


# velocity of the red ball
red_ball_velocity = velocity(start_depth_image, final_depth_image, rxcntr1, rycntr1, rxcntr2, rycntr2)
brown_ball_velocity = velocity(start_depth_image, final_depth_image, bxcntr1, bycntr1, bxcntr2, bycntr2)
print("velocity of the red ball:", red_ball_velocity)

# velocity of the brown ball
print("velocity of the brown ball:", brown_ball_velocity)

# relative velocity of balls
print("Relative velocity", red_ball_velocity + brown_ball_velocity)

while True:
    cv2.imshow("Color Depth Image 1", start_depth_real)
    cv2.imshow("Color Depth Image 2", final_depth_real)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
