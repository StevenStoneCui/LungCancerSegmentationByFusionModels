# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 14:30:43 2021

@author: ariken
"""

import os

path = 'save\\'
for x in os.listdir(path):
    if os.path.isdir(os.path.join(path, x)):
        for y in os.listdir(os.path.join(path, x)):
            for z in os.listdir(os.path.join(path, x, y)):
                os.remove(os.path.join(path, x, y, z))
            