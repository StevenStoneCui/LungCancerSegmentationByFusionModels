# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:26:23 2021

@author: ariken
"""
from __future__ import absolute_import, print_function
import cupy as cp

class RandomFlipLayer(object):
    """
    Add a random flipping layer as pre-processing.
    """

    def __init__(self,
                 flip_axes = (1,),
                 flip_probability=0.5,
                 name='random_flip'):
        """

        :param flip_axes: a list of indices over which to flip
        :param flip_probability: the probability of performing the flip
            (default = 0.5)
        :param name:
        """
        self._flip_axes = flip_axes
        self._flip_probability = flip_probability
        self._rand_flip = None
        self.randomise()


    def randomise(self, spatial_rank=3):
        spatial_rank = int(cp.floor(spatial_rank))
        self._rand_flip = cp.random.random(
            size=spatial_rank) < self._flip_probability


    def _apply_transformation(self, image):
        assert self._rand_flip is not None, "Flip is unset -- Error!"
        for axis_number, do_flip in enumerate(self._rand_flip):
            if axis_number in self._flip_axes and do_flip:
                image = cp.flip(image, axis=axis_number)
        return image

    def layer_op(self, inputs):
        if inputs is None:
            return inputs
        result = []        
        for i in inputs:
            result.append(self._apply_transformation(i))
        return result