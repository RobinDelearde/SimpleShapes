#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Generate SimpleShapes images/datasets
    Generate (datasets of) images with random shapes, from some pre-defined shapes or from shapes given as images.
    
    1. SimpleShape generators: Generate shapes from a pre-defined list, with random scale and scales of parts if specified.
    2. Image from shapes generator: Make an image from a list of pre-defined shapes or from a list of image shapes given as input,
       with random rotations and translations, with or without overlapping.
    3. Image from random shapes generator: Generate images containing a given number of random shapes from a list, with random parameters, and also generate:
          - the separated image of each shape
          - for shapes from generators: the image of exploded shapes with parts at random position and with random orientation
    4. Dataset of random shapes generator: Generate datasets of images containing a given number of random shapes from a list, with random parameters.
    
    author: R. Deléarde
    initial version @CNAM/CEDRIC: 25/10/2019
    extended version @LIPADE/SIP: 23/03/2020-07/04/2020
"""

#%% Part 1: code

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from basic_shapes import generate_flat_rectangle, generate_flat_triangle, generate_flat_rect_triangle, generate_semiellipse, \
                        generate_small_rectangle, generate_small_triangle, generate_small_ellipse, generate_small_semiellipse
from simple_shapes import generate_house, generate_tree, generate_boat, generate_plane, generate_jet, \
                            generate_rocket, generate_submarine, generate_tractor, generate_car, generate_truck

empty_array = np.array([], dtype=np.uint8)

###############################################################################
# rotate shape inside the image
###############################################################################
from math import cos, sin, pi

#first solution: around center of wrapping rectangle, handmade (pb: lose points)
def apply_rotations(rr, cc, angle, x_max = 128, y_max = 128): #rotation around the computed center of the shape (angle in rad)
    rr_center, cc_center = (min(rr)+max(rr))//2, (min(cc)+max(cc))//2
    rr_centered, cc_centered = rr-rr_center, cc-cc_center
    rr_rotated = np.rint(rr_centered*cos(angle) - cc_centered*sin(angle)) + rr_center
    cc_rotated = np.rint(rr_centered*sin(angle) + cc_centered*cos(angle)) + cc_center
    rr_rotated = rr_rotated.astype(int)
    cc_rotated = cc_rotated.astype(int)
    return rr_rotated, cc_rotated

#second solution: around shape center, handmade (pb: lose points, and shape center is not really central for all shapes => not used)
#def apply_rotations2(shape, angle, x_max = 128, y_max = 128): #rotation around the shape center attribute (angle in rad)
#    rr_center, cc_center = shape.x, shape.y
#    rr, cc = shape.coords()
#    rr_centered, cc_centered = rr-rr_center, cc-cc_center
#    rr_rotated = np.rint(rr_centered*cos(angle) - cc_centered*sin(angle)) + rr_center
#    cc_rotated = np.rint(rr_centered*sin(angle) + cc_centered*cos(angle)) + cc_center
#    rr_rotated = rr_rotated.astype(int)
#    cc_rotated = cc_rotated.astype(int)
#    return rr_rotated, cc_rotated

# third solution: around center of wrapping rectangle, with skimage.transform.rotate + check with apply_rotations
def rotate_shape(rr, cc, angle=0, x_max=128, y_max=128):
    assert(min(rr)>=0 and max(rr)<x_max and min(cc)>=0 and max(cc)<y_max), "input shape outside image"
    if angle == 0:
        return rr, cc
    else:
        # test if rotated image is out of bounds (require to use apply_rotations instead of rotate because rotate eliminates outbound values)
        rr1, cc1 = apply_rotations(rr, cc, angle*pi/180, x_max, y_max)
        if (min(rr1)<0 or max(rr1)>=x_max or min(cc1)<0 or max(cc1)>=y_max):
            return empty_array, empty_array
        else:
            # apply rotation with rotate
            img = np.zeros((x_max, y_max), dtype=np.uint8)
            img[rr,cc] = 1
            rr_center, cc_center = (min(rr)+max(rr))//2, (min(cc)+max(cc))//2
            img1 = np.ceil(rotate(img, angle, center = (cc_center, rr_center)))
            rr1, cc1 = np.nonzero(img1)
            return rr1, cc1

"""usage:
rr, cc, rr_parts, cc_parts = generate_plane(x_max=256, y_max=256)
img = np.zeros((256, 256), dtype=np.uint8)
img[rr, cc] = 1
plt.imshow(img) and plt.show()
rr_rotated, cc_rotated = apply_rotations(rr, cc, -30*pi/180, 256, 256)
img = np.zeros((256, 256), dtype=np.uint8)
img[rr_rotated, cc_rotated] = 1
plt.imshow(img) and plt.show()
rr_rotated, cc_rotated = rotate_shape(rr, cc, -30, 256, 256)
img = np.zeros((256, 256), dtype=np.uint8)
img[rr_rotated, cc_rotated] = 1
plt.imshow(img) and plt.show()
"""

###############################################################################
# generate images
###############################################################################

# rotate the shape with random angle inside the image
def rotate_shape_random(rr, cc, x_max=128, y_max=128, random=None, verbose=False, max_trials=10):
    if (not rr.size): #empty shape
        print("try to rotate empty shape")
        return rr, cc
    else:
        if random == None:
            random = np.random.RandomState()
        rr1, nb_trials = empty_array, 0
        while (not rr1.size) and (nb_trials<max_trials):
            nb_trials+=1
            angle = random.randn(1)*360
            rr1, cc1 = rotate_shape(rr, cc, angle, x_max, y_max)
        if verbose and (not rr1.size):
            print("could not rotate the shape inside the image")
        return rr1, cc1
"""usage:
rr_rotated, cc_rotated = rotate_shape_random(rr, cc, 256, 256, None, True, 5)
img = np.zeros((256, 256), dtype=np.uint8)
img[rr_rotated, cc_rotated] = 1
plt.imshow(img) and plt.show()
"""

# translate a shape inside the image (NB: input shape must be inside the image)
def translate_shape_random(rr, cc, x_max=128, y_max=128):
    if (not rr.size): #empty shape
        print("try to translate empty shape")
        return rr, cc
    else:
        rr_min, rr_max, cc_min, cc_max = min(rr), max(rr), min(cc), max(cc)
        assert(rr_min>=0 and rr_max<x_max and cc_min>=0 and cc_max<y_max), "input shape outside image"
        x_move = np.random.randint(-rr_min, x_max-rr_max)
        y_move = np.random.randint(-cc_min, y_max-cc_max)
        return rr+x_move, cc+y_move
"""usage:
rr_translated, cc_translated = translate_shape_random(rr, cc, 256, 256)
img = np.zeros((256, 256), dtype=np.uint8)
img[rr_translated, cc_translated] = 1
plt.imshow(img) and plt.show()
"""

# generate shape with random parameters (unused)
#def generate_shape(generate_shape_function, scale=None, variance=0.4, x_max=128, y_max=128, random=None):
#    rr, cc = generate_shape_function(x_max, y_max, scale, variance)
#    rr, cc = rotate_shape_random(rr, cc, x_max, y_max, random)
#    rr, cc = translate_shape_random(rr, cc, x_max, y_max)
#    return rr, cc

from skimage.morphology import binary_dilation
selem = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]])

# add one shape to an image, test several orientations and positions if required
def add_shape(img, rr, cc, use_rotations=True, allow_overlap=False, random=None, verbose=False, max_trials=20, max_overlap_rate=0.2):
    if (not rr.size):
        print("try to insert empty shape")
        job_done = True
    else:
        x_max, y_max = img.shape
        if random == None:
            random = np.random.RandomState()
        job_done = False
        nb_trials = 0
        while (not job_done) and nb_trials<max_trials:
            nb_trials+=1
            if use_rotations:
                rr1, cc1 = rotate_shape_random(rr, cc, x_max, y_max, random, verbose)
            else:
                rr1, cc1 = rr, cc
            if rr1.size: #test should always be ok when this function is called by add_random_shape because of assert tests
                rr1, cc1 = translate_shape_random(rr1, cc1, x_max, y_max)
                img1 = np.zeros((x_max, y_max), dtype=np.uint8)
                img1[rr1,cc1] = 1
                if allow_overlap: # NB: no dilation used here
                    img_surf = img.sum()
                    img1_surf = img1.sum()
                    overlap = (img*img1).sum()
                    min_surf = min(img_surf, img1_surf)
                    if min_surf==0 or (overlap/min_surf < max_overlap_rate):
                        img[rr1,cc1] = 1 #img=max(img,img1)
                        job_done = True
                else: # NB: with dilation to avoid too close shapes, if not wanted, delete or use allow_overlap with max_overlap_rate=0.
                    img1d = np.uint8(binary_dilation(img1, selem))
                    overlap = (img*img1d).max()
                    if overlap == 0:
                        img[rr1,cc1] = 1 #img=max(img,img1)
                        job_done = True
        if verbose:
            print('nb trials:', nb_trials)
            #if not job_done:
            #    print("could not insert the shape")
    return img, job_done, img1
"""usage:
rr, cc, rr_parts, cc_parts = generate_plane(x_max=256, y_max=256)
img = np.zeros((256, 256), dtype=np.uint8)
img, job_done, img1 = add_shape(img, rr, cc, True, False, None, True, 10)
plt.imshow(img) and plt.show()
plt.imshow(img1) and plt.show()
img, job_done, img1 = add_shape(img, rr, cc, True, True, None, True, 10, 0.2)
plt.imshow(img) and plt.show()
plt.imshow(img1) and plt.show()
"""

### with SimpleShapes generators

# add one random shape (from a list) to an image and its parts to another image
# scale: scale factor of recommended scale
# variance: factor of recommended variance (variance=0 => no variance)
def add_random_shape(img, img_parts, shape_generators_list, scale=1, variance=1, \
                     use_rotations=True, allow_overlap=False, random=None, verbose=False, max_trials=20, max_overlap_rate=0.2):
    assert(img_parts.shape == img.shape)
    x_max, y_max = img.shape
    if random == None:
        random = np.random.RandomState()
    shape_nr = random.randint(len(shape_generators_list))
    generate_shape_function = shape_generators_list[shape_nr]
    # generate and insert the simple shape
    if verbose:
        print("insert shape:", generate_shape_function)
        input_ok, nb_trials = False, 0
    while not input_ok and nb_trials<2: # only 2 tries to generate a correct input, if it doesn't succeed then it's better to modify the parameters of the generate_[shape] function
        nb_trials+=1
        rr, cc, rr_parts, cc_parts = generate_shape_function(x_max, y_max, scale, variance)
        input_ok = rr.size and min(rr)>=0 and max(rr)<x_max and min(cc)>=0 and max(cc)<y_max
        if not input_ok:
            print("random shape outside image limits")
    assert(input_ok), "couldn't generate random shape inside image limits, consider to modify parameters of the generate_[shape] function"
    img, job_done, img1 = add_shape(img, rr, cc, use_rotations, allow_overlap, random, verbose, max_trials, max_overlap_rate) #TODO: set 2 different overlap parameters for entire/exploded shape
    assert(job_done), "could not insert the simple shape (not enough space left in image)"
    # insert the parts
    if verbose:
        print("insert parts of the shape")
    job2_done, nb_trials = False, 0
    while not job2_done and nb_trials<max_trials: #max_trials to insert all the parts (and not only the last one as in add_shape)
        nb_trials+=1
        img_parts_i = np.array(img_parts, copy=True)
        for i in range(len(rr_parts)):
            img_parts_i, job2_done, _ = add_shape(img_parts_i, rr_parts[i], cc_parts[i], use_rotations, allow_overlap, random, False, max_trials, max_overlap_rate) #max_trials for each part
    if not job2_done:
        print('trial nr %d not succeeded' %nb_trials)
        plt.imshow(img_parts_i) and plt.show()
    assert(job2_done), "could not insert the parts (not enough space left in image)"
    if verbose:
        print('nb trials:', nb_trials)
    return img, img_parts_i, img1
"""usage:
shapes_list = {0: generate_truck}
img = np.zeros((256, 256), dtype=np.uint8)
img_parts = np.zeros((256, 256), dtype=np.uint8)
img, img_parts, img1 = add_random_shape(img, img_parts, shapes_list, 0.5, 1, True, False, None, True, 10)
plt.imshow(img) and plt.show()
plt.imshow(img_parts) and plt.show()
img, img_parts, img1 = add_random_shape(img, img_parts, shapes_list, 0.5, 1, True, False, None, True, 10)
plt.imshow(img) and plt.show()
plt.imshow(img_parts) and plt.show()
plt.imshow(img1) and plt.show()
"""

# generate an image containing several random shapes
def generate_image(shape_generators_list=None, nb_shapes=2, x_max=128, y_max=128, scale=1, variance=1, \
                    use_rotations=True, allow_overlap=False, random_seed=None, verbose=False, max_trials=20, max_overlap_rate=0.2):
    img = np.zeros((x_max, y_max), dtype=np.uint8)
    img_parts = np.zeros((x_max, y_max), dtype=np.uint8)
    random = np.random.RandomState(random_seed)
    img_list = []
    for i in range(nb_shapes):
        img, img_parts, img1 = add_random_shape(img, img_parts, shape_generators_list, scale, variance, use_rotations, allow_overlap, random, verbose, max_trials, max_overlap_rate)
        img_list.append(img1)
    return img, img_parts, img_list
"""usage:
shapes_list = {0: generate_house,
               1: generate_tree,
               2: generate_boat,
               3: generate_plane,
               4: generate_jet,
               5: generate_rocket,
               6: generate_submarine,
               7: generate_tractor,
               8: generate_car,
               9: generate_truck}
img, img_parts, img_list = generate_image(shapes_list, 2, 256, 256, 0.5, 1, True, False, None, True, 20)
plt.imshow(img) and plt.show()
plt.imshow(img_parts) and plt.show()
plt.imshow(img_list[0]) and plt.show()
plt.imshow(img_list[1]) and plt.show()
"""

# generate an image containing some given shapes + several random shapes
def generate_image2(imposed_shapes_list=None, random_shapes_list=None, nb_random_shapes=2, x_max=128, y_max=128, scale=1, variance=1, \
                     use_rotations=True, allow_overlap=False, random_seed=None, verbose=False, max_trials=20, max_overlap_rate=0.2):
    img = np.zeros((x_max, y_max), dtype=np.uint8)
    img_parts = np.zeros((x_max, y_max), dtype=np.uint8)
    random = np.random.RandomState(random_seed)
    img_list = []
    for i in range(len(imposed_shapes_list)):
        img, img_parts, img1 = add_random_shape(img, img_parts, [imposed_shapes_list[i]], scale, variance, use_rotations, allow_overlap, random, verbose, max_trials, max_overlap_rate)
        img_list.append(img1)
    for i in range(nb_random_shapes):
        img, img_parts, img1 = add_random_shape(img, img_parts, random_shapes_list, scale, variance, use_rotations, allow_overlap, random, verbose, max_trials, max_overlap_rate)
        img_list.append(img1)
    return img, img_parts, img_list
"""usage:
shapes_list1 = {0: generate_jet}
shapes_list2 = {0: generate_tree,
                1: generate_house}
img, img_parts, img_list = generate_image2(shapes_list1, shapes_list2, 2, 256, 256, 0.5, 1, True, False, None, True, 20)
plt.imshow(img) and plt.show()
plt.imshow(img_parts) and plt.show()
plt.imshow(img_list[0]) and plt.show()
plt.imshow(img_list[1]) and plt.show()
plt.imshow(img_list[2]) and plt.show()
"""

### from images

# make an image with given shape images
# shape_list: list of binary/grayscale 2D images, each containing 1 shape
def image_from_shapes(shape_images_list=None, x_max=128, y_max=128, use_rotations=True, allow_overlap=False, random_seed=None, verbose=False, max_trials=20, max_overlap_rate=0.2):
    img = np.zeros((x_max, y_max), dtype=np.uint8)
    random = np.random.RandomState(random_seed)
    img_list = []
    for shape_img in shape_images_list:
        if len(shape_img.shape)>2:
            shape_img = shape_img[:,:,0]
        shape_points = np.argwhere(shape_img>0)
        rr, cc = shape_points[:,0], shape_points[:,1]
        input_ok = max(rr)-min(rr)<x_max and max(cc)-min(cc)<y_max
        assert(input_ok), "input shape too large for specified image size"
        # move the shape at the center of the new image (required because input shape must be inside image for the rotation and the translation)
        x_move = (x_max-min(rr)-max(rr))//2
        y_move = (y_max-min(cc)-max(cc))//2
        rr += x_move
        cc += y_move
        # insert shape at random position, and with random orientation if required
        img, job_done, img1 = add_shape(img, rr, cc, use_rotations, allow_overlap, random, verbose, max_trials, max_overlap_rate)
        assert(job_done), "could not insert the shape (not enough space left in image)"
        #TODO: move the previous shape if don't succeed to insert the next one ? better to use larger output image size
        img_list.append(img1)
    return img, img_list
"""usage:
shape1 = plt.imread("img1.png")
shape2 = plt.imread("img2.png")
shape1 = plt.imread("AS02N003.pgm")
shape2 = plt.imread("AS06N002.pgm")
shape_images_list = [shape1, shape2]
img, img_list = image_from_shapes(shape_images_list, 256, 256, True, False, None, True, 20)
plt.imshow(img) and plt.show()
plt.imshow(img_list[0]) and plt.show()
plt.imshow(img_list[1]) and plt.show()
"""

###############################################################################
# generate datasets
###############################################################################
import os
from PIL import Image
#import matplotlib.cm as cm

### with SimpleShapes generators

# generate several images containing several random shapes obtained with random generator
def generate_dataset(shape_generators_list=None, nb_images=1, output_dir="", nb_shapes=2, x_max=128, y_max=128, scale=1, variance=1, \
                    use_rotations=True, allow_overlap=False, random_seed=None, verbose=False, max_trials=20, max_overlap_rate=0.2):
    if not ((output_dir=="") or output_dir[len(output_dir)-1]=='/'):
        output_dir = output_dir+'/'
    os.makedirs(output_dir+'/full', exist_ok=True)
    os.makedirs(output_dir+'/exploded', exist_ok=True)
    os.makedirs(output_dir+'/separated', exist_ok=True)
    for i in range(nb_images):
        if verbose:
            print('****** image nr %d ******' %(i+1))
        img, img_parts, img_list = generate_image(shape_generators_list, nb_shapes, x_max, y_max, scale, variance, \
                    use_rotations, allow_overlap, random_seed, verbose, max_trials, max_overlap_rate)
        img_pil = Image.fromarray(255*img)
        img_pil.save(output_dir+'full/img%d.png' %(i+1)) # PNG with 1 channel
        #plt.imsave(output_dir+'full/img%d.png' %(i+1), img, cmap=cm.gray) # PNG with 4 identical channels
        img_parts_pil = Image.fromarray(255*img_parts)
        img_parts_pil.save(output_dir+'exploded/img_parts_%d.png' %(i+1)) # PNG with 1 channel
        #plt.imsave(output_dir+'exploded/img_parts_%d.png' %(i+1), img_parts, cmap=cm.gray) # PNG with 4 identical channels
        for j, img in enumerate(img_list):
            #plt.imsave(output_dir+'separated/img%d_%d.png' %(i+1, j+1), img, cmap=cm.gray) # 4 channels
            img_pil = Image.fromarray(255*img)
            img_pil.save(output_dir+'separated/img%d_%d.png' %(i+1, j+1)) # PNG with 1 channel

# generate several images containing some given shapes + several random shapes obtained with random generator
def generate_dataset2(imposed_shapes_list=None, random_shapes_list=None, nb_images=1, output_dir="", nb_random_shapes=2, x_max=128, y_max=128, \
                      scale=1, variance=1, use_rotations=True, allow_overlap=False, random_seed=None, verbose=False, max_trials=20, max_overlap_rate=0.2):
    if not ((output_dir=="") or output_dir[len(output_dir)-1]=='/'):
        output_dir = output_dir+'/'
    os.makedirs(output_dir+'/full', exist_ok=True)
    os.makedirs(output_dir+'/exploded', exist_ok=True)
    os.makedirs(output_dir+'/separated', exist_ok=True)
    for i in range(nb_images):
        if verbose:
            print('****** image nr %d ******' %(i+1))
        img, img_parts, img_list = generate_image2(imposed_shapes_list, random_shapes_list, nb_random_shapes, x_max, y_max, scale, variance, \
                                         use_rotations, allow_overlap, random_seed, verbose, max_trials, max_overlap_rate)
        img_pil = Image.fromarray(255*img)
        img_pil.save(output_dir+'full/img%d.png' %(i+1)) # PNG with 1 channel
        #plt.imsave(output_dir+'full/img%d.png' %(i+1), img, cmap=cm.gray) # PNG with 4 identical channels
        img_parts_pil = Image.fromarray(255*img_parts)
        img_parts_pil.save(output_dir+'exploded/img_parts_%d.png' %(i+1)) # PNG with 1 channel
        #plt.imsave(output_dir+'exploded/img_parts_%d.png' %(i+1), img_parts, cmap=cm.gray) # PNG with 4 identical channels
        for j, img in enumerate(img_list):
            #plt.imsave(output_dir+'separated/img%d_%d.png' %(i+1, j+1), img, cmap=cm.gray) # PNG with 4 identical channels
            img_pil = Image.fromarray(255*img)
            img_pil.save(output_dir+'separated/img%d_%d.png' %(i+1, j+1)) # PNG with 1 channel

# generate dataset of shapes pairs from random shape generators
def generate_2shapes_dataset_old(shape_generators_list=None, nb_images=1, output_dir="", x_max=128, y_max=128, scale=0.5, variance=1, \
                    use_rotations=True, allow_overlap=False, random_seed=None, verbose=False, all_pairs=False, max_trials=20, max_overlap_rate=0.2):
    n = len(shape_generators_list)
    assert(n>1)
    if all_pairs:
        for i in range(n):
            for j in range(n):
                generate_dataset2({0: shape_generators_list[i], 1: shape_generators_list[j]}, {}, nb_images, output_dir+'/'+str(i+1)+'-'+str(j+1), \
                                  0, x_max, y_max, scale, variance, use_rotations, allow_overlap, random_seed, verbose, max_trials, max_overlap_rate)
    else:
        for i in range(n):
            for j in range(i+1,n):
                generate_dataset2({0: shape_generators_list[i], 1: shape_generators_list[j]}, {}, nb_images, output_dir+'/'+str(i+1)+'-'+str(j+1), \
                                  0, x_max, y_max, scale, variance, use_rotations, allow_overlap, random_seed, verbose, max_trials, max_overlap_rate)

# generate dataset of shapes pairs from random shape generators
def generate_2shapes_dataset(shape_generators_list=None, nb_images=1, output_dir="", x_max=128, y_max=128, scale=0.5, variance=1, \
                    use_rotations=True, allow_overlap=False, random_seed=None, verbose=False, all_pairs=False, max_trials=20, max_overlap_rate=0.2):
    if not ((output_dir=="") or output_dir[len(output_dir)-1]=='/'):
        output_dir = output_dir+'/'
    os.makedirs(output_dir+'/full', exist_ok=True)
    os.makedirs(output_dir+'/exploded', exist_ok=True)
    os.makedirs(output_dir+'/separated', exist_ok=True)
    n = len(shape_generators_list)
    assert(n>1)
    for i in range(n):
        j_start = 0 if all_pairs else i+1
        for j in range(j_start, n):
            for k in range(nb_images):
                if verbose:
                    print('****** image nr %d ******' %(k+1))
                img, img_parts, img_list = generate_image2({0: shape_generators_list[i], 1: shape_generators_list[j]}, {}, 0, x_max, y_max, \
                        scale, variance, use_rotations, allow_overlap, random_seed, verbose, max_trials, max_overlap_rate)
                img_pil = Image.fromarray(255*img)
                img_pil.save(output_dir+'full/img-%d-%d-%d.png' %(i+1,j+1,k+1)) # PNG with 1 channel
                #plt.imsave(output_dir+'full/img-%d-%d-%d.png' %(i+1,j+1,k+1), img, cmap=cm.gray) # PNG with 4 identical channels
                img_parts_pil = Image.fromarray(255*img_parts)
                img_parts_pil.save(output_dir+'exploded/img_parts-%d-%d-%d.png' %(i+1,j+1,k+1)) # PNG with 1 channel
                #plt.imsave(output_dir+'exploded/img_parts-%d-%d-%d.png' %(i+1,j+1,k+1), img_parts, cmap=cm.gray) # PNG with 4 identical channels
                for l, img in enumerate(img_list):
                    #plt.imsave(output_dir+'separated/img-%d-%d-%d-%d.png' %(i+1,j+1,k+1,l+1), img, cmap=cm.gray) # PNG with 4 identical channels
                    img_pil = Image.fromarray(255*img)
                    img_pil.save(output_dir+'separated/img-%d-%d-%d-%d.png' %(i+1,j+1,k+1,l+1)) # PNG with 1 channel

"""
usage:
generate_dataset(shapes_list, 10, "dataset", 2, 256, 256, 0.6, 0.8, verbose=True)
generate_dataset2(shapes_list1, shapes_list2, 10, "dataset", 1, 256, 256, 0.6, 0.8, verbose=True)
generate_dataset2(shapes_list2, {}, 10, "house-tree", 0, 256, 256, 0.5, 1, verbose=True)
generate_2shapes_dataset(shapes_list, 10, "dataset", 256, 256, 0.5, 1, verbose=True)
"""

### from images

# generate dataset from given images of shapes, with random positions and orientations (if specified)
def generate_dataset_from_images(shape_images_list=None, nb_images=1, output_dir="", x_max=128, y_max=128, \
                    use_rotations=True, allow_overlap=False, random_seed=None, verbose=False, max_trials=20, max_overlap_rate=0.2):
    if not ((output_dir=="") or output_dir[len(output_dir)-1]=='/'):
        output_dir = output_dir+'/'
    os.makedirs(output_dir+'/full', exist_ok=True)
    os.makedirs(output_dir+'/separated', exist_ok=True)
    for i in range(nb_images):
        if verbose:
            print('****** image nr %d ******' %(i+1))
        img, img_list = image_from_shapes(shape_images_list, x_max, y_max, use_rotations, allow_overlap, random_seed, verbose, max_trials, max_overlap_rate)
        img_pil = Image.fromarray(255*img)
        img_pil.save(output_dir+'full/img%d.png' %(i+1)) # PNG with 1 channel
        #plt.imsave(output_dir+'full/img%d.png' %(i+1), img, cmap=cm.gray) # PNG with 4 identical channels
        for j, img_j in enumerate(img_list):
            #plt.imsave(output_dir+'separated/img%d_%d.png' %(i+1, j+1), img, cmap=cm.gray) # PNG with 4 identical channels
            img_pil = Image.fromarray(255*img_j)
            img_pil.save(output_dir+'separated/img%d_%d.png' %(i+1, j+1)) # PNG with 1 channel

# generate dataset of shapes pairs from given images of shapes, with random positions and orientations (if specified)
def generate_2shapes_dataset_from_images(shape_images_list=None, nb_images=1, output_dir="", x_max=128, y_max=128, \
                    use_rotations=True, allow_overlap=False, random_seed=None, verbose=False, max_trials=20, max_overlap_rate=0.2, all_pairs=False):
    n = len(shape_images_list)
    assert(n>1)
    if all_pairs:
        for i in range(n):
            #j_start = 0 if all_pairs else i+1
            for j in range(n):
                generate_dataset_from_images([shape_images_list[i], shape_images_list[j]], nb_images, output_dir+'/'+str(i+1)+'-'+str(j+1), \
                        x_max, y_max, use_rotations, allow_overlap, random_seed, verbose, max_trials, max_overlap_rate)
    else:
        for i in range(n):
            #j_start = 0 if all_pairs else i+1
            for j in range(i+1,n):
                generate_dataset_from_images([shape_images_list[i], shape_images_list[j]], nb_images, output_dir+'/'+str(i+1)+'-'+str(j+1), \
                        x_max, y_max, use_rotations, allow_overlap, random_seed, verbose, max_trials, max_overlap_rate)

"""
usage:
shape1 = plt.imread("img1.png")
shape2 = plt.imread("img2.png")
shape3 = plt.imread("AS02N003.pgm")
shape4 = plt.imread("AS06N002.pgm")
shape_images_list = [shape1, shape2, shape3, shape4]
generate_dataset_from_images(shape_images_list, 10, "dataset", 320, 320, allow_overlap=True, verbose=True)
generate_2shapes_dataset_from_images(shape_images_list, 10, "dataset", 320, 320, verbose=True, all_pairs=False)
"""

#%% Part 2: generation of the datasets

# generate dataset1bis (full) and dataset2bis (parts)
generate_dataset({0: generate_house}, 1000, "datasets_v2/house", 1, x_max=256, y_max=256, verbose=True)
generate_dataset({0: generate_tree}, 1000, "datasets_v2/tree", 1, x_max=256, y_max=256, verbose=True)
generate_dataset({0: generate_plane}, 1000, "datasets_v2/plane", 1, x_max=256, y_max=256, verbose=True)
generate_dataset({0: generate_jet}, 1000, "datasets_v2/jet", 1, x_max=256, y_max=256, verbose=True)
generate_dataset({0: generate_rocket}, 1000, "datasets_v2/rocket", 1, x_max=256, y_max=256, verbose=True)
generate_dataset({0: generate_boat}, 1000, "datasets_v2/boat", 1, x_max=256, y_max=256, verbose=True)
generate_dataset({0: generate_submarine}, 1000, "datasets_v2/submarine", 1, x_max=256, y_max=256, verbose=True)
generate_dataset({0: generate_tractor}, 1000, "datasets_v2/tractor", 1, x_max=256, y_max=256, verbose=True)
generate_dataset({0: generate_car}, 1000, "datasets_v2/car", 1, x_max=256, y_max=256, verbose=True)
generate_dataset({0: generate_truck}, 1000, "datasets_v2/truck", 1, x_max=256, y_max=256, verbose=True)

# generate dataset1ter (full) and dataset2ter (parts with overlapping)
generate_dataset({0: generate_house}, 1000, "datasets_v3/house", 1, x_max=256, y_max=256, allow_overlap=True, verbose=True)
generate_dataset({0: generate_tree}, 1000, "datasets_v3/tree", 1, x_max=256, y_max=256, allow_overlap=True, verbose=True)
generate_dataset({0: generate_plane}, 1000, "datasets_v3/plane", 1, x_max=256, y_max=256, allow_overlap=True, verbose=True)
generate_dataset({0: generate_jet}, 1000, "datasets_v3/jet", 1, x_max=256, y_max=256, allow_overlap=True, verbose=True)
generate_dataset({0: generate_rocket}, 1000, "datasets_v3/rocket", 1, x_max=256, y_max=256, allow_overlap=True, verbose=True)
generate_dataset({0: generate_boat}, 1000, "datasets_v3/boat", 1, x_max=256, y_max=256, allow_overlap=True, verbose=True)
generate_dataset({0: generate_submarine}, 1000, "datasets_v3/submarine", 1, x_max=256, y_max=256, allow_overlap=True, verbose=True)
generate_dataset({0: generate_tractor}, 1000, "datasets_v3/tractor", 1, x_max=256, y_max=256, allow_overlap=True, verbose=True)
generate_dataset({0: generate_car}, 1000, "datasets_v3/car", 1, x_max=256, y_max=256, allow_overlap=True, verbose=True)
generate_dataset({0: generate_truck}, 1000, "datasets_v3/truck", 1, x_max=256, y_max=256, allow_overlap=True, verbose=True)

# generate dataset of shapes pairs for simple shapes
shapes_list = {0: generate_house, 1: generate_tree, 2: generate_boat, 3: generate_plane, 4: generate_jet, \
               5: generate_rocket, 6: generate_submarine, 7: generate_tractor, 8: generate_car, 9: generate_truck}
generate_2shapes_dataset(shapes_list, 10, "dataset", x_max=224, y_max=224, scale=0.5, variance=1, verbose=True, all_pairs=True)

# generate dataset of shapes pairs for basic shapes (rectangle, ellipse, triangle)
shapes_list = {0: generate_flat_rectangle, 1: generate_flat_triangle, 2: generate_flat_rect_triangle, 3: generate_semiellipse, \
               4: generate_small_rectangle, 5: generate_small_ellipse, 6: generate_small_triangle, 7: generate_small_semiellipse}
generate_2shapes_dataset(shapes_list, 20, "dataset", x_max=224, y_max=224, scale=1, variance=1, verbose=True, all_pairs=True)
