import numpy as np
import cv2

def rgb_to_gray(img):
    """
    Convert to grayscale
    """
    return  np.sum(img/3, axis=3, keepdims=True)

def normalize(img):
    """
    Normalize the image
    """
    return (img - 128)/128

def translate(img):
    """
    Translate the image by upto 3 pixels in a random direction
    """
    rows,cols,_ = img.shape
    # Using openCV for image translation: 
    # https://docs.opencv.org/3.1.0/da/d6e/tutorial_py_geometric_transformations.html
    px = 2
    t_x, t_y = np.random.randint(-px,px,2)
    M = np.float32([[1,0,t_x],[0,1,t_y]])     
    return cv2.warpAffine(img,M,(cols,rows))    

def scale(img):  
    rows,cols,_ = img.shape

    # Create a random pixel number between -4 and 4
    px = np.random.randint(-4,4)

    # points for perspective transform
    pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]]) 
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(img,M,(rows,cols))

def increase_brightness(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    br = 0.5+np.random.uniform()
    if br > 1.0:
        br = 1.0
    img[:,:,2] = img[:,:,2]*br
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return img

def rotate(img):
    """
    Rotate the image by a random angle theta between 0 to 2pi
    """
    cx,cy = int(img.shape[0]/2), int(img.shape[1]/2)
    angle = 30.0*np.random.rand()-12
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    return cv2.warpAffine(img, M, img.shape[:2]), angle

def sharpen(img):
    gauss_blur = cv2.GaussianBlur(img, (5,5), 20.0)
    return cv2.addWeighted(img, 2, gauss_blur, -1, 0)

def augmented_data(X_train):
    """
    This function creates 3 different arrays: 
    X_train_rot: array of rotated images
    X_train_scaled: array of scaled images
    X_train_trans_sharp: array of translated and sharpened images
    """
    rotated_arr = []
    scaled_arr = []
    trans_sharp_arr = []
    for i in range(len(X_train)):
        rotated_img, _ = rotate(X_train[i])
        scaled_img = scale(X_train[i])
        trans_sharp_img = translate(sharpen(X_train[i]))
        rotated_arr.append(rotated_img)
        scaled_arr.append(scaled_img)
        trans_sharp_arr.append(trans_sharp_img)

    X_train_rot = np.concatenate([arr[np.newaxis] for arr in rotated_arr])
    X_train_scaled = np.concatenate([arr[np.newaxis] for arr in scaled_arr])
    X_train_trans_sharp = np.concatenate([arr[np.newaxis] for arr in trans_sharp_arr])
    
    return X_train_rot, X_train_scaled, X_train_trans_sharp

    
    
