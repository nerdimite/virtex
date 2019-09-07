import cv2 as cv
import numpy as np

def chk_typ(obj, req_type):
    return req_type == type(obj)

def grid_coords(resolution,image):

    '''
    RETURNS grid coordinates and grid pixel's height and width

    PARAMETERS:
    resolution (int) :- The number of grid pixels along on direction(horizontally or vertically)
                       This parameter increases the sensitivity
    image :- The image on which the grid needs to be applied
    '''

    height = image.shape[0]
    width = image.shape[1]

    grid_x = np.linspace(0,width,int(resolution+1))
    grid_y = np.linspace(0,height,int(resolution+1))
    gpixel_width = grid_x[1]-grid_x[0]
    gpixel_height = grid_y[1]-grid_y[0]

    return grid_x, grid_y, gpixel_width, gpixel_height

def grid_pixel_search(x,y,resolution,gpixel_width,gpixel_height):

    '''
    RETURNS the pixel number counted from left to right starting
    from top-right grid pixel of the image

    PARAMETERS:
    x (int) :- x coordinate of the point
    y (int) :- y coordinate of the point
    resolution (int) :- The number of grid pixels along on direction(horizontally or vertically).
                  This parameter increases the sensitivity.
    gpixel_width (float) :- Width of each grid pixel obtained through the grid_coords function
    gpixel_height (float) :- Height of each grid pixel obtained through the grid_coords function
    '''

    xb = int(x/gpixel_width)
    yb = int(y/gpixel_height)

    gpixel = ((yb * int(resolution)) + xb)

    return gpixel

def draw_grid(image, grid_x, grid_y):

    '''
    RETURNS image array with the approximate grid drawn for visual purposes

    PARAMETERS:
    image :- The image on which the grid needs to be drawn
    grid_x (list) :- List containing the x coordinates of the grid
    grid_y (list) :- List containing the y coordinates of the grid
    '''

    for xp in grid_x:
        cv.line(image, (int(round(xp)), 0), (int(round(xp)), image.shape[0]), (0, 0, 255), 1, 1)
    for yp in grid_y:
        cv.line(image, (0, int(round(yp))), (image.shape[1], int(round(yp))), (0, 0, 255), 1, 1)
