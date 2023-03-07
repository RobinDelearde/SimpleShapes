#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to copy images in a new directory sorting them according to their level annotation.

author: R. Deléarde @LIPADE (09/04/2020)
"""

import os

#%%
image_dir='' # specify here or use current directory in Spyder
separator = ' '

os.mkdir(image_dir+'N1')
os.mkdir(image_dir+'N2')
os.mkdir(image_dir+'N3')
os.mkdir(image_dir+'N4')

annotation_file = image_dir+'annotations.csv'
with open(annotation_file) as f:
    for line in f:
        line = line[0:len(line)-1].split(separator) # len(line)-1 to remove the '\n'
        os.system('cp '+line[0]+' N'+line[2])
