#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Classes of basic geometric shapes
    
    Elements are defined by a center, a scale (on x and y).
    Since the 2D coordinates in numpy are line-column, this means that x is actually the height
    and y is the width. We invert (x,y) to be less confusing when drawing figures.
    However y increases when going "down" in the image.
    
    author: R. Deléarde
    initial version @CNAM/CEDRIC: 15/10/2019
        based on N. Audebert @CNAM/CEDRIC (10/2019), https://github.com/nshaud/shapes
    extended version @LIPADE/SIP: 07/04/2020
"""
import numpy as np
from skimage import draw
import matplotlib.pyplot as plt #useful for usage tests

################
### RECTANGLES

class Rectangle():
    def __init__(self, center, width, height=None):
        self.y, self.x = center
        self.width = width
        self.height = height if height is not None else width
    
    def coords(self):
        top_left = self.x - (self.height // 2), self.y - (self.width // 2)
        #bottom_right = self.x + (self.height // 2), self.y + (self.width // 2)
        rr,cc = draw.rectangle(top_left, extent=(self.height, self.width))
        return (np.reshape(rr,rr.size),np.reshape(cc,cc.size))

def generate_rectangle(x_max=128, y_max=128, scale=1, variance=0):
    scale = scale*0.5*min(x_max, y_max)
    variance = 0.4*variance
    width = np.random.randint(scale-variance*scale, scale+variance*scale+1)
    height = np.random.randint(width-variance*width, width+variance*width+1)//2
    rr, cc = Rectangle((x_max//2, y_max//2), width, height).coords()
    return rr, cc, [rr], [cc]

def generate_flat_rectangle(x_max=128, y_max=128, scale=1, variance=0):
    scale = scale*0.6*min(x_max, y_max)
    variance = 0.5*variance
    width = np.random.randint(scale-0.5*variance*scale, scale+0.5*variance*scale+1)
    height = np.random.randint(width-variance*width, width+variance*width+1)//16
    rr, cc = Rectangle((x_max//2, y_max//2), width, height).coords()
    return rr, cc, [rr], [cc]

def generate_small_rectangle(x_max=128, y_max=128, scale=1, variance=0):
    return generate_rectangle(x_max=x_max, y_max=y_max, scale=0.2*scale, variance=variance)

################
### ELLIPSES

class Ellipse():
    def __init__(self, center, y_axis, x_axis=None, intern_radius=0):
        self.y, self.x = center
        self.y_axis = y_axis
        self.x_axis = x_axis if x_axis is not None else y_axis
        self.intern_radius = intern_radius
    
    def coords(self):
        rr, cc = draw.ellipse(self.x, self.y, self.y_axis, self.x_axis)
        if self.intern_radius>0:
            #mask = ((cc-self.y)/self.x_axis)**2+((rr-self.x)/self.y_axis)**2>=self.intern_radius
            #return (rr*mask, cc*mask) #pb: all masked points are now (0,0)
            value = ((cc-self.y)/self.x_axis)**2+((rr-self.x)/self.y_axis)**2
            pos = np.nonzero(value>=self.intern_radius)[0]
            return (rr[pos], cc[pos])
        else:
            return rr, cc

def generate_ellipse(x_max=128, y_max=128, scale=1, variance=0):
    scale = scale*0.28*min(x_max, y_max)
    x_axis = np.random.randint(scale-0.2*variance*scale, scale+0.2*variance*scale+1)
    y_axis = np.random.randint(x_axis-0.4*variance*x_axis, x_axis+0.4*variance*x_axis+1)*2//5
    rr, cc = Ellipse((x_max//2, y_max//2), y_axis, x_axis).coords()
    return rr, cc, [rr], [cc]

def generate_small_ellipse(x_max=128, y_max=128, scale=1, variance=0):
    return generate_ellipse(x_max=x_max, y_max=y_max, scale=0.4*scale, variance=0.8*variance)

class SemiEllipse():
    def __init__(self, center, y_axis, x_axis=None, hidden_part=0.5, intern_radius=0):
        self.y, self.x = center
        self.y_axis = y_axis
        self.x_axis = x_axis if x_axis is not None else y_axis
        self.hidden_part = hidden_part
        self.intern_radius = intern_radius
    
    def coords(self):
        rr, cc = draw.ellipse(self.x, self.y, self.y_axis, self.x_axis)
        # mask = cc<=self.y+(1-2*self.hidden_part)*self.x_axis
        # if self.intern_radius>0:
        #     mask2 = ((cc-self.y)/self.x_axis)**2+((rr-self.x)/self.y_axis)**2>=self.intern_radius
        #     mask = mask*mask2
        # return (rr*mask, cc*mask) #pb: all masked points are now (0,0)
        pos = np.nonzero(cc<=self.y+(1-2*self.hidden_part)*self.x_axis)[0]
        rr = rr[pos]
        cc = cc[pos]
        if self.intern_radius>0:
            value = ((cc-self.y)/self.x_axis)**2+((rr-self.x)/self.y_axis)**2
            pos = np.nonzero(value>=self.intern_radius)[0]
            return rr[pos], cc[pos]
        else:
            return rr, cc

def generate_semiellipse(x_max=128, y_max=128, scale=1, variance=0):
    scale = scale*0.28*min(x_max, y_max)
    x_axis = np.random.randint(scale-0.2*variance*scale, scale+0.2*variance*scale+1)
    y_axis = np.random.randint(x_axis-0.2*variance*x_axis, x_axis+0.2*variance*x_axis+1)
    hidden_part = 0.5*(1+np.random.rand(1)*0.2)
    intern_radius = 0.5*(1+np.random.rand(1)*0.2)
    rr, cc = SemiEllipse((x_max//2, y_max//2), y_axis, x_axis, hidden_part, intern_radius).coords()
    return rr, cc, [rr], [cc]

def generate_small_semiellipse(x_max=128, y_max=128, scale=1, variance=0):
    scale = scale*0.08*min(x_max, y_max)
    x_axis = np.random.randint(scale-0.2*variance*scale, scale+0.2*variance*scale+1)
    y_axis = np.random.randint(x_axis-0.2*variance*x_axis, x_axis+0.2*variance*x_axis+1)
    hidden_part = 0.4*(1+np.random.rand(1)*0.2)
    rr, cc = SemiEllipse((x_max//2, y_max//2), y_axis, x_axis, hidden_part).coords()
    return rr, cc, [rr], [cc]

################
### TRIANGLES

class Triangle():
    def __init__(self, top, left_offset, right_offset):
        self.y, self.x = top
        self.left_y, self.left_x = left_offset
        self.right_y, self.right_x = right_offset
    
    def coords(self):
        rows = [self.x, self.x + self.left_x, self.x + self.right_x]
        columns = [self.y, self.y + self.left_y, self.y + self.right_y]
        coords = draw.polygon(rows, columns)
        return coords

def generate_triangle(x_max=128, y_max=128, scale=1, variance=0):
    scale = scale*0.4*min(x_max, y_max)
    variance = 0.4*variance
    width = np.random.randint(scale-variance*scale, scale+variance*scale+1)
    height = np.random.randint(width-variance*width, width+variance*width+1)*2//3
    top_delta_x = np.random.randint(-variance*width, variance*width+1)//2
    rr, cc = Triangle((x_max//2+top_delta_x, (y_max-height)//2), (-width//2-top_delta_x, height), (width//2-top_delta_x, height)).coords()
    return rr, cc, [rr], [cc]

def generate_small_triangle(x_max=128, y_max=128, scale=1, variance=0):
    return generate_triangle(x_max=x_max, y_max=y_max, scale=0.6*scale, variance=0.6*variance)

def generate_rect_triangle(x_max=128, y_max=128, scale=1, variance=0):
    scale = scale*0.4*min(x_max, y_max)
    variance = 0.4*variance
    width = np.random.randint(scale-variance*scale, scale+variance*scale+1)
    height = np.random.randint(width-variance*width, width+variance*width+1)
    rr, cc = Triangle(((x_max-width)//2+width//8, (y_max-height)//2), (0, height), (width, height)).coords()
    return rr, cc, [rr], [cc]

def generate_flat_triangle(x_max=128, y_max=128, scale=1, variance=0):
    scale = scale*0.7*min(x_max, y_max)
    variance = 0.3*variance
    width = np.random.randint(scale-variance*scale, scale+variance*scale+1)
    height = np.random.randint(width-0.5*variance*width, width+0.5*variance*width+1)//8
    top_delta_x = np.random.randint(-variance*width, variance*width+1)//2
    rr, cc = Triangle((x_max//2+top_delta_x, (y_max-height)//2), (-width//2-top_delta_x, height), (width//2-top_delta_x, height)).coords()
    return rr, cc, [rr], [cc]

def generate_flat_rect_triangle(x_max=128, y_max=128, scale=1, variance=0):
    scale = scale*0.7*min(x_max, y_max)
    variance = 0.2*variance
    width = np.random.randint(scale-variance*scale, scale+variance*scale+1)
    height = np.random.randint(width-variance*width, width+variance*width+1)//7
    rr, cc = Triangle(((x_max-width)//2+width//16, (y_max-height)//2), (0, height), (width, height)).coords()
    return rr, cc, [rr], [cc]

"""usage:
# Draw samples of each shape
img = np.zeros((128, 128), dtype=np.uint8)

rectangle = Rectangle((64, 64), 10, 30)
img[rectangle.coords()] = 1

ellipse = Ellipse((64, 64), 10, 40, 0.5)
img[ellipse.coords()] = 1

semiellipse = SemiEllipse((64, 64), 20, 40, 0.4, 0.6)
img[semiellipse.coords()] = 1

triangle = Triangle((5, 5), (0, 30), (30, 30))
img[triangle.coords()] = 1

plt.imshow(img) and plt.show()
"""

"""usage of generate_[shape]:
rr, cc, _, _ = generate_[shape](128, 128, 1, 1)

img = np.zeros((128, 128), dtype=np.uint8)
img[rr, cc] = 1
plt.imshow(img) and plt.show()
"""
