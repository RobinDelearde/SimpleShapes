#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""" Classes of simple shapes (formed by a set of basic shapes)
    + generate_[shape] functions with recommended scale and variance for each shape
    coords() and generate_[shape] functions generate 2 shape outputs (rr, cc):
        - one for the simple shape with concatenated parts and
        - one with separated parts in a list (useful to generate the image of exploded shapes).
    
    Elements are defined by a center and a scale (on x and y).
    Since the 2D coordinates in numpy are line-column, this means that x is actually the height
    and y is the width. We invert (x,y) to be less confusing when drawing figures.
    However y increases when going "down" in the image.
    
    author: R. Deléarde @CNAM/CEDRIC (25/10/2019)
"""

import numpy as np
from basic_shapes import Rectangle, Ellipse, Triangle
import matplotlib.pyplot as plt #useful for usage tests

empty_output = np.array([], dtype=np.uint8), np.array([], dtype=np.uint8), [], []
# useful for the triangles that don't record negative values in the output shape => manual check: return empty_output when a vertex is <0

class House():
    def __init__(self, center, scale, variance=0):
        self.x, self.y = center #center of the wall
        self.scale = scale #in pixels
        self.variance = variance
    
    def coords(self): #NB: change at every call because of random parameters (unless variance=0)
        # parameters
        width = np.random.randint(self.scale-self.variance*self.scale, self.scale+self.variance*self.scale+1)
        wall_height = np.random.randint(self.scale-0.5*self.variance*self.scale, self.scale+0.5*self.variance*self.scale+1)
        roof_height = np.random.randint(wall_height-self.variance*wall_height, wall_height+1)//2
        rooftop = self.y-wall_height//2-roof_height
        if rooftop<0:
            return empty_output
        else:
            # create objects
            wall = Rectangle((self.x, self.y), width, wall_height)
            roof = Triangle((self.x, rooftop), (-width//2, roof_height), (width//2, roof_height))
            # compute parts coordinates
            rr1, cc1 = wall.coords()
            rr2, cc2 = roof.coords()
            # decomposition in object parts
            rr_parts = [rr1, rr2]
            cc_parts = [cc1, cc2]
            # concatenate the row and column coordinates
            rr = np.concatenate((rr1, rr2))
            cc = np.concatenate((cc1, cc2))
            return rr, cc, rr_parts, cc_parts

def generate_house(x_max=128, y_max=128, scale=1, variance=0):
    return House((x_max//2, int(y_max*0.6)), scale*0.46*min(x_max, y_max), 0.3*variance).coords()

"""usage:
house = House((128, 200), 80, 0.4)
rr, cc, rr_parts, cc_parts = house.coords()
img = np.zeros((256, 256), dtype=np.uint8)
img[rr, cc] = 1
plt.imshow(img) and plt.show()
"""

class Tree():
    def __init__(self, center, scale, variance=0):
        self.x, self.y = center #center of the foliage
        self.scale = scale #in pixels
        self.variance = variance
    
    def coords(self):
        # paramaters
        trunk_height = np.random.randint(self.scale-self.variance*self.scale, self.scale+self.variance*self.scale+1)
        foliage_height = np.random.randint(trunk_height-self.variance*trunk_height, trunk_height+self.variance*trunk_height+1)
        foliage_width = np.random.randint(foliage_height//2, foliage_height+1)
        trunk_width = np.random.randint(foliage_width-self.variance*foliage_width, foliage_width+self.variance*foliage_width+1)//2
        # create objects
        trunk = Rectangle((self.x, self.y+trunk_height//2+foliage_height), trunk_width, trunk_height+foliage_height//16)
        foliage = Ellipse((self.x, self.y), foliage_height, foliage_width)
        # compute parts coordinates
        rr1, cc1 = trunk.coords()
        rr2, cc2 = foliage.coords()
        # prepare outputs
        rr_parts = [rr1, rr2]
        cc_parts = [cc1, cc2]
        rr = np.concatenate((rr1, rr2))
        cc = np.concatenate((cc1, cc2))
        return rr, cc, rr_parts, cc_parts

def generate_tree(x_max=128, y_max=128, scale=1, variance=0):
    return Tree((x_max//2, int(y_max*0.4)), scale*0.24*min(x_max, y_max), 0.14*variance).coords()

"""usage:
tree = Tree((128, 100), 60, 0.2)
rr, cc, rr_parts, cc_parts = tree.coords()
img = np.zeros((256, 256), dtype=np.uint8)
img[rr, cc] = 1
plt.imshow(img) and plt.show()
"""

class Boat():
    def __init__(self, center, scale, variance=0):
        self.x, self.y = center #center of the hull
        self.scale = scale
        self.variance = variance
    
    def coords(self):
        # paramaters
        hull_width = np.random.randint(self.scale-self.variance*self.scale, self.scale+self.variance*self.scale+1)
        mast_height = np.random.randint(hull_width-1.6*self.variance*hull_width, hull_width+1.6*self.variance*hull_width+1)
        hull_height = np.random.randint(mast_height-1.6*self.variance*mast_height, mast_height+1.6*self.variance*mast_height+1)//4
        mast_width =  np.random.randint(mast_height-2*self.variance*mast_height, mast_height+2*self.variance*mast_height+1)//16
        # create objects
        hull = Rectangle((self.x, self.y), hull_width, hull_height)
        mast = Rectangle((self.x, self.y-mast_height//2-hull_height//2), mast_width, mast_height)
        top1 = (self.x+mast_width//2, self.y-mast_height-hull_height//2)
        top2 = (self.x-mast_width, self.y-mast_height-hull_height//2)
        sail1 = Triangle(top1, (0, mast_height-mast_height//16), (hull_width//2-mast_width//2, mast_height-mast_height//16))
        sail2 = Triangle(top2, (0, mast_height-mast_height//8), (-hull_width//2+mast_width, mast_height-mast_height//8))
        # compute parts coordinates
        rr1, cc1 = hull.coords()
        rr2, cc2 = mast.coords()
        rr3, cc3 = sail1.coords()
        rr4, cc4 = sail2.coords()
        # prepare outputs
        rr_parts = [np.concatenate((rr2, rr3)), rr1, rr4]
        cc_parts = [np.concatenate((cc2, cc3)), cc1, cc4]
        rr = np.concatenate((rr1, rr2, rr3, rr4))
        cc = np.concatenate((cc1, cc2, cc3, cc4))
        return rr, cc, rr_parts, cc_parts

def generate_boat(x_max=128, y_max=128, scale=1, variance=0):
    return Boat((x_max//2, int(y_max*0.85)), scale*0.56*min(x_max, y_max), 0.1*variance).coords()

"""usage:
boat = Boat((128, 200), 120, 0.35)
rr, cc, rr_parts, cc_parts = boat.coords()
img = np.zeros((256, 256), dtype=np.uint8)
img[rr, cc] = 1
plt.imshow(img) and plt.show()
"""

class Plane():
    def __init__(self, center, scale, variance=0):
        self.x, self.y = center
        self.scale = scale
        self.variance = variance
    
    def coords(self):
        # paramaters
        body_lenght = np.random.randint(self.scale-self.variance*self.scale, self.scale+self.variance*self.scale+1)
        body_widht = np.random.randint(body_lenght-0.8*self.variance*body_lenght, body_lenght+0.8*self.variance*body_lenght+1)//5
        wings_length = np.random.randint((body_lenght-0.6*self.variance*body_lenght)*2, (body_lenght+0.6*self.variance*body_lenght+1)*2)
        wings_width = np.random.randint(wings_length-0.8*self.variance*wings_length, wings_length+0.8*self.variance*wings_length+1)//6
        wings_center = np.random.randint(self.scale-0.5*self.variance*self.scale, self.scale+0.5*self.variance*self.scale+1)//8
        tailplane_length = np.random.randint(wings_length-0.8*self.variance*wings_length, wings_length+0.8*self.variance*wings_length+1)//4
        tailplane_width = np.random.randint(tailplane_length-0.8*self.variance*tailplane_length, tailplane_length+0.8*self.variance*tailplane_length+1)//3
        # create objects
        body = Ellipse((self.x, self.y), body_lenght, body_widht)
        wings = Rectangle((self.x, self.y-wings_center), wings_length, wings_width)
        tailplane = Rectangle((self.x, self.y+5*body_lenght//6), tailplane_length, tailplane_width)
        # compute parts coordinates
        rr1, cc1 = body.coords()
        rr2, cc2 = wings.coords()
        rr3, cc3 = tailplane.coords()
        # prepare outputs
        rr_parts = [np.concatenate((rr1, rr3)), rr2]
        cc_parts = [np.concatenate((cc1, cc3)), cc2]
        rr = np.concatenate((rr1, rr2, rr3))
        cc = np.concatenate((cc1, cc2, cc3))
        return rr, cc, rr_parts, cc_parts

def generate_plane(x_max=128, y_max=128, scale=1, variance=0):
    return Plane((x_max//2, y_max//2), scale*0.36*min(x_max, y_max), 0.2*variance).coords()

"""usage:
plane = Plane((128, 128), 90, 0.2)
rr, cc, rr_parts, cc_parts = plane.coords()
img = np.zeros((256, 256), dtype=np.uint8)
img[rr, cc] = 1
plt.imshow(img) and plt.show()
"""

class Jet():
    def __init__(self, center, scale, variance=0):
        self.x, self.y = center
        self.scale = scale
        self.variance = variance
    
    def coords(self):
        # paramaters
        body_lenght = np.random.randint(self.scale-self.variance*self.scale, self.scale+self.variance*self.scale+1)
        body_widht = np.random.randint(body_lenght-0.4*self.variance*body_lenght, body_lenght+0.4*self.variance*body_lenght+1)//5
        wings_width = np.random.randint((body_lenght-0.4*self.variance*body_lenght)*4, (body_lenght+0.4*self.variance*body_lenght+1)*4)//5
        wings_top = np.random.randint(body_lenght-0.6*self.variance*body_lenght, body_lenght+0.6*self.variance*body_lenght+1)//2
        if self.x-wings_width<0:
            return empty_output
        else:
            # create objects
            body = Ellipse((self.x, self.y), body_lenght, body_widht)
            wings = Triangle((self.x, self.y-wings_top), (-wings_width, body_lenght+wings_top*2//3), (wings_width, body_lenght+wings_top*2//3))
            # compute parts coordinates
            rr1, cc1 = wings.coords()
            rr2, cc2 = body.coords()
            # prepare outputs
            rr_parts = [rr1, rr2]
            cc_parts = [cc1, cc2]
            rr = np.concatenate((rr1, rr2))
            cc = np.concatenate((cc1, cc2))
            return rr, cc, rr_parts, cc_parts

def generate_jet(x_max=128, y_max=128, scale=1, variance=0):
    return Jet((x_max//2, y_max//2), scale*0.38*min(x_max, y_max), 0.2*variance).coords()

"""usage:
jet = Jet((128, 128), 90, 0.3)
rr, cc, rr_parts, cc_parts = jet.coords()
img = np.zeros((256, 256), dtype=np.uint8)
img[rr, cc] = 1
plt.imshow(img) and plt.show()
"""

class Rocket():
    def __init__(self, center, scale, variance=0):
        self.x, self.y = center
        self.scale = scale
        self.variance = variance
    
    def coords(self):
        # paramaters
        body_lenght = np.random.randint(self.scale-self.variance*self.scale, self.scale+self.variance*self.scale+1)
        body_widht = np.random.randint(body_lenght-0.6*self.variance*body_lenght, body_lenght+0.6*self.variance*body_lenght+1)//5
        wings_width = np.random.randint(body_lenght-0.4*self.variance*body_lenght, body_lenght+0.4*self.variance*body_lenght+1)//2
        wings_top = np.random.randint(body_lenght-0.4*self.variance*body_lenght, body_lenght+0.4*self.variance*body_lenght+1)//5
        wings_bottom = np.random.randint(body_lenght-0.4*self.variance*body_lenght, body_lenght+0.4*self.variance*body_lenght+1)//2
        if self.x-wings_width<0:
            return empty_output
        else:
            # create objects
            body = Ellipse((self.x, self.y), body_lenght, body_widht)
            wings = Triangle((self.x, self.y+wings_top), (-wings_width, wings_bottom), (wings_width, wings_bottom))
            # compute parts coordinates
            rr1, cc1 = body.coords()
            rr2, cc2 = wings.coords()
            # prepare outputs
            rr_parts = [rr1, rr2]
            cc_parts = [cc1, cc2]
            rr = np.concatenate((rr1, rr2))
            cc = np.concatenate((cc1, cc2))
            return rr, cc, rr_parts, cc_parts

def generate_rocket(x_max=128, y_max=128, scale=1, variance=0):
    return Rocket((x_max//2, y_max//2), scale*0.38*min(x_max, y_max), 0.3*variance).coords()

"""usage:
rocket = Rocket((128, 128), 90, 0.3)
rr, cc, rr_parts, cc_parts = rocket.coords()
img = np.zeros((256, 256), dtype=np.uint8)
img[rr, cc] = 1
plt.imshow(img) and plt.show()
"""

class Submarine():
    def __init__(self, center, scale, variance=0):
        self.x, self.y = center
        self.scale = scale
        self.variance = variance
    
    def coords(self):
        # paramaters
        body_lenght = np.random.randint(self.scale-self.variance*self.scale, self.scale+self.variance*self.scale+1)
        body_height = np.random.randint(body_lenght-self.variance*body_lenght, body_lenght+self.variance*body_lenght+1)//4
        tower_length = np.random.randint((body_lenght-0.6*self.variance*body_lenght)*2, (body_lenght+0.6*self.variance*body_lenght+1)*2)//5
        tower_height = np.random.randint((body_height-0.6*self.variance*body_height)*3, (body_height+0.6*self.variance*body_height+1)*3)//2
        tower_position = np.random.randint(-self.variance*body_lenght, self.variance*body_lenght+1)
        # create objects
        body = Ellipse((self.x, self.y), body_height, body_lenght)
        tower = Rectangle((self.x+tower_position, self.y-body_height-tower_height//2+(tower_position//10)**2), tower_length, tower_height)
        # compute parts coordinates
        rr1, cc1 = body.coords()
        rr2, cc2 = tower.coords()
        # prepare outputs
        rr_parts = [rr1, rr2]
        cc_parts = [cc1, cc2]
        rr = np.concatenate((rr1, rr2))
        cc = np.concatenate((cc1, cc2))
        return rr, cc, rr_parts, cc_parts

def generate_submarine(x_max=128, y_max=128, scale=1, variance=0):
    return Submarine((x_max//2, y_max//2), scale*0.4*min(x_max, y_max), 0.2*variance).coords()

"""usage:
submarine = Submarine((128, 128), 90, 0.3)
rr, cc, rr_parts, cc_parts = submarine.coords()
img = np.zeros((256, 256), dtype=np.uint8)
img[rr, cc] = 1
plt.imshow(img) and plt.show()
"""

class Tractor():
    def __init__(self, center, scale, variance=0):
        self.x, self.y = center
        self.scale = scale
        self.variance = variance
    
    def coords(self):
        # paramaters
        cockpit_height = np.random.randint(self.scale-self.variance*self.scale, self.scale+self.variance*self.scale+1)
        cockpit_length = np.random.randint((cockpit_height-0.5*self.variance*cockpit_height)*3, (cockpit_height+0.5*self.variance*cockpit_height)*3+1)//4
        engine_length = np.random.randint((cockpit_length-0.5*self.variance*cockpit_length)*7, (cockpit_length+0.5*self.variance*cockpit_length)*7+1)//8
        engine_height = np.random.randint(cockpit_height-0.6*self.variance*cockpit_height, cockpit_height+0.6*self.variance*cockpit_height+1)//2
        mid_total_length = (cockpit_length+engine_length)//2
        tractor_dim = max(mid_total_length, cockpit_height) # =cockpit_height most of the time since total_length=7/8*(1+3/4)*cockpit_height+random
        wheel_radius = np.random.randint(tractor_dim-0.6*self.variance*tractor_dim, tractor_dim+0.6*self.variance*tractor_dim+1)//4
        # create objects
        cockpit = Rectangle((self.x, self.y), cockpit_length, cockpit_height)
        engine = Rectangle((self.x+mid_total_length, self.y+cockpit_height//2-engine_height//2), engine_length, engine_height)
        wheel1 = Ellipse((self.x-mid_total_length//8, self.y+cockpit_height//2), wheel_radius, wheel_radius)
        wheel2 = Ellipse((self.x+mid_total_length, self.y+cockpit_height//2), wheel_radius, wheel_radius)
        # compute parts coordinates
        rr1, cc1 = cockpit.coords()
        rr2, cc2 = engine.coords()
        rr3, cc3 = wheel1.coords()
        rr4, cc4 = wheel2.coords()
        # prepare outputs
        rr_parts = [np.concatenate((rr1, rr2)), rr3, rr4]
        cc_parts = [np.concatenate((cc1, cc2)), cc3, cc4]
        rr = np.concatenate((rr1, rr2, rr3, rr4))
        cc = np.concatenate((cc1, cc2, cc3, cc4))
        return rr, cc, rr_parts, cc_parts

def generate_tractor(x_max=128, y_max=128, scale=1, variance=0):
    return Tractor((int(x_max*0.35), y_max//2), scale*0.48*min(x_max, y_max), 0.14*variance).coords()

"""usage:
tractor = Tractor((90, 128), 110, 0.25)
rr, cc, rr_parts, cc_parts = tractor.coords()
img = np.zeros((256, 256), dtype=np.uint8)
img[rr, cc] = 1
plt.imshow(img) and plt.show()
"""

class Car():
    def __init__(self, center, scale, variance=0):
        self.x, self.y = center
        self.scale = scale
        self.variance = variance
    
    def coords(self):
        # paramaters
        cockpit_height = np.random.randint(self.scale-self.variance*self.scale, self.scale+self.variance*self.scale+1)
        cockpit_length = np.random.randint((cockpit_height-0.5*self.variance*cockpit_height)*5, (cockpit_height+0.5*self.variance*cockpit_height)*5+1)//4
        engine_length = np.random.randint((cockpit_length-0.5*self.variance*cockpit_length)*3, (cockpit_length+0.5*self.variance*cockpit_length)*3+1)//4
        engine_height = np.random.randint(cockpit_height-0.5*self.variance*cockpit_height, cockpit_height+0.5*self.variance*cockpit_height+1)//2
        car_dim = max((cockpit_length+engine_length)*2//5, cockpit_height) # =first term most of the time since cockpit_length+engine_length)=21/16*cockpit_height+random
        wheel_radius = np.random.randint(car_dim-0.4*self.variance*car_dim, car_dim+0.4*self.variance*car_dim+1)//4
        # create objects
        cockpit = Rectangle((self.x, self.y), cockpit_length, cockpit_height)
        engine = Rectangle((self.x+(cockpit_length+engine_length)//2, self.y+cockpit_height//2-engine_height//2), engine_length, engine_height)
        front = Triangle((self.x+cockpit_length//2, self.y-cockpit_height//2), (0, cockpit_height-engine_height), (engine_length//2, cockpit_height-engine_height))
        wheel1 = Ellipse((self.x-cockpit_length//4, self.y+cockpit_height//2), wheel_radius, wheel_radius)
        wheel2 = Ellipse((self.x+cockpit_length//2+5*engine_length//8, self.y+cockpit_height//2), wheel_radius, wheel_radius)
        # compute parts coordinates
        rr1, cc1 = cockpit.coords()
        rr2, cc2 = engine.coords()
        rr3, cc3 = front.coords()
        rr4, cc4 = wheel1.coords()
        rr5, cc5 = wheel2.coords()
        # prepare outputs
        rr_parts = [np.concatenate((rr1, rr2, rr3)), rr4, rr5]
        cc_parts = [np.concatenate((cc1, cc2, cc3)), cc4, cc5]
        rr = np.concatenate((rr1, rr2, rr3, rr4, rr5))
        cc = np.concatenate((cc1, cc2, cc3, cc4, cc5))
        return rr, cc, rr_parts, cc_parts

def generate_car(x_max=128, y_max=128, scale=1, variance=0):
    return Car((int(x_max*0.31), y_max//2), scale*0.36*min(x_max, y_max), 0.16*variance).coords()

"""usage:
car = Car((80, 128), 76, 0.2)
rr, cc, rr_parts, cc_parts = car.coords()
img = np.zeros((256, 256), dtype=np.uint8)
img[rr, cc] = 1
plt.imshow(img) and plt.show()
"""

class Truck():
    def __init__(self, center, scale, variance=0):
        self.x, self.y = center
        self.scale = scale
        self.variance = variance
    
    def coords(self):
        # paramaters
        container_height = np.random.randint(self.scale-self.variance*self.scale, self.scale+self.variance*self.scale+1)
        container_length = np.random.randint((container_height-1.2*self.variance*container_height)*2, (container_height+1.2*self.variance*container_height)*2+1)
        cockpit_height = np.random.randint((container_height-0.8*self.variance*container_height)*7, (container_height+0.8*self.variance*container_height)*7+1)//8
        cockpit_length = np.random.randint((container_height-0.8*self.variance*container_height)*3, (container_height+0.8*self.variance*container_height)*3+1)//4
        truck_dim = max(container_length*4//11, cockpit_height)
        wheel_radius = np.random.randint(truck_dim-0.5*self.variance*truck_dim, truck_dim+0.5*self.variance*truck_dim+1)//4
        # create objects
        container = Rectangle((self.x, self.y), container_length, container_height)
        cockpit = Rectangle((self.x+(container_length+cockpit_length)//2+int(self.scale/32), self.y+container_height//2-cockpit_height//2), cockpit_length, cockpit_height)
        wheel1 = Ellipse((self.x-3*container_length//10, self.y+container_height//2), wheel_radius, wheel_radius)
        wheel2 = Ellipse((self.x-3*container_length//10+2*wheel_radius, self.y+container_height//2), wheel_radius, wheel_radius)
        wheel3 = Ellipse((self.x+container_length//2+cockpit_length//2-2*wheel_radius, self.y+container_height//2), wheel_radius, wheel_radius)
        wheel4 = Ellipse((self.x+container_length//2+cockpit_length//2, self.y+container_height//2), wheel_radius, wheel_radius)
        # compute parts coordinates
        rr1, cc1 = container.coords()
        rr2, cc2 = cockpit.coords()
        rr3, cc3 = wheel1.coords()
        rr4, cc4 = wheel2.coords()
        rr5, cc5 = wheel3.coords()
        rr6, cc6 = wheel4.coords()
        # prepare outputs
        rr_parts = [rr1, rr2, rr3, rr4, rr5, rr6]
        cc_parts = [cc1, cc2, cc3, cc4, cc5, cc6]
        rr = np.concatenate((rr1, rr2, rr3, rr4, rr5, rr6))
        cc = np.concatenate((cc1, cc2, cc3, cc4, cc5, cc6))
        return rr, cc, rr_parts, cc_parts

def generate_truck(x_max=128, y_max=128, scale=1, variance=0):
    return Truck((int(x_max*0.4), y_max//2), scale*0.28*min(x_max, y_max), 0.11*variance).coords()

"""usage:
truck = Truck((100, 128), 70, 0.1)
rr, cc, rr_parts, cc_parts = truck.coords()
img = np.zeros((256, 256), dtype=np.uint8)
img[rr, cc] = 1
plt.imshow(img) and plt.show()
"""

class Lshape():
    def __init__(self, center, scale, variance=0):
        self.x, self.y = center #center of the long part
        self.scale = scale #in pixels
        self.variance = variance
    
    def coords(self):
        # parameters
        height = np.random.randint(self.scale-self.variance*self.scale, self.scale+self.variance*self.scale+1)
        length = np.random.randint(height-0.5*self.variance*height, height+0.5*self.variance*height+1)//3
        width = np.random.randint(height-self.variance*height, height+self.variance*height+1)//8
        # create objects
        long_part = Rectangle((self.x, self.y), width, height)
        short_part = Rectangle((self.x+length//2, self.y+(height-width)//2), length, width)
        # compute parts coordinates
        rr1, cc1 = long_part.coords()
        rr2, cc2 = short_part.coords()
        # decomposition in object parts
        rr_parts = [rr1, rr2]
        cc_parts = [cc1, cc2]
        # concatenate the row and column coordinates
        rr = np.concatenate((rr1, rr2))
        cc = np.concatenate((cc1, cc2))
        return rr, cc, rr_parts, cc_parts

def generate_Lshape(x_max=128, y_max=128, scale=1, variance=0):
    return Lshape((x_max//2, y_max//2), scale*0.44*min(x_max, y_max), 0.4*variance).coords()

"""usage:
lshape = Lshape((128, 128), 80, 0.4)
rr, cc, rr_parts, cc_parts = lshape.coords()
img = np.zeros((256, 256), dtype=np.uint8)
img[rr, cc] = 1
plt.imshow(img) and plt.show()
"""

class Tshape():
    def __init__(self, center, scale, variance=0):
        self.x, self.y = center #center of the long part
        self.scale = scale #in pixels
        self.variance = variance
    
    def coords(self):
        # parameters
        height = np.random.randint(self.scale-self.variance*self.scale, self.scale+self.variance*self.scale+1)
        length = np.random.randint(height-0.2*self.variance*height, height+0.8*self.variance*height+1)//2
        width = np.random.randint(height-self.variance*height, height+self.variance*height+1)//8
        # create objects
        long_part = Rectangle((self.x, self.y), width, height)
        short_part = Rectangle((self.x, self.y-height//2+width//2), length, width)
        # compute parts coordinates
        rr1, cc1 = long_part.coords()
        rr2, cc2 = short_part.coords()
        # decomposition in object parts
        rr_parts = [rr1, rr2]
        cc_parts = [cc1, cc2]
        # concatenate the row and column coordinates
        rr = np.concatenate((rr1, rr2))
        cc = np.concatenate((cc1, cc2))
        return rr, cc, rr_parts, cc_parts

def generate_Tshape(x_max=128, y_max=128, scale=1, variance=0):
    return Tshape((x_max//2, y_max//2), scale*0.44*min(x_max, y_max), 0.3*variance).coords()

"""usage:
tshape = Tshape((128, 128), 80, 0.4)
rr, cc, rr_parts, cc_parts = tshape.coords()
img = np.zeros((256, 256), dtype=np.uint8)
img[rr, cc] = 1
plt.imshow(img) and plt.show()
"""

"""usage of generate_[shape]:
rr, cc, rr_parts, cc_parts = generate_[shape](256, 256, 1, 1)

img = np.zeros((256, 256), dtype=np.uint8)
img[rr, cc] = 1
plt.imshow(img) and plt.show()

img = np.zeros((256, 256), dtype=np.uint8)
img[rr_parts[0], cc_parts[0]] = 1
plt.imshow(img) and plt.show()
"""
