#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to annotate a dataset with spatial relations, using keys 2-4-6-8
of the numeric keypad or keys N-H-K-U, and ? for ambiguous cases.

author: R. Deléarde @LIPADE (08/04/2020)
"""

import os
# import sys
from matplotlib import pyplot as plt

# from terminal with command 'python annotator.py [my_image_dir]'
#image_dir = sys.argv[1]
#files_list = sorted(os.listdir(image_dir))

# from Spyder with image directory = current directory
image_dir=''
files_list = sorted(os.listdir())

print('******************************************************* \n' + \
      'position of the green shape relative to the yellow one: \n' + \
      '(4/H = left, 8/U = up, 6/K = right, 2/N = down, ? = ?)  \n' + \
      '******************************************************* \n')
list_correct = ['4', '8', '6', '2', 'H', 'U', 'K', 'N', '?']
for f in files_list:
    img_path = image_dir+f
    #if os.isfile(img_path):
    img = plt.imread(img_path)
    plt.imshow(img) and plt.show()
    annotation_file = open(image_dir+'annotations.csv','a')
    annotation_ok = False
    while not annotation_ok:
        annotation = input(img_path+' : ')
        if any(annotation==s for s in list_correct):
            annotation_ok = True
        else:
            question_ok = False
            while not question_ok:
                question = input('You wrote : '+annotation+'\n' +\
                                 'Is it ok ? (Y: yes, N: no) \n')
                if question=='N':
                    question_ok = True
                elif question=='Y':
                    question_ok = True
                    annotation_ok = True
    annotation_file.write(img_path+' '+annotation+"\n")
    annotation_file.close()
