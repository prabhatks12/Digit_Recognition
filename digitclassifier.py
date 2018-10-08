# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 22:16:11 2018

@author: prabhat
"""

import matplotlib.pyplot as py
from sklearn import datasets
from PIL import Image


from sklearn import svm

digit = datasets.load_digits()
 
img=svm.SVC()
x,y=digit.data[:],digit.target[:]
img.fit(x,y)

#i is the image to be predicted
i=7

print('Predicted Value:',img.predict(digit.data[i]))

im=py.imshow(digit.images[i],cmap=py.cm.gray_r,interpolation='nearest')

#saves the image of predicted picture with black background

Image.fromarray(digit.images[i]).convert('LA').resize((150, 300)).save('data.png')
