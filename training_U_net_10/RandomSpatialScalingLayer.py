# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:08:01 2021

@author: ariken
"""
from __future__ import absolute_import, print_function


import cupy as cp
import cupyx.scipy.ndimage as cpndi

class RandomSpatialScalingLayer(object):
    """
    generate randomised scaling along each dim for data augmentation
    """

    def __init__(self,
                 min_percentage=-10.0,
                 max_percentage=10.0,
                 antialiasing=True,
                 isotropic=False,
                 name='random_spatial_scaling'):
        assert min_percentage <= max_percentage
        self._min_percentage = max(min_percentage, -99.9)
        self._max_percentage = max_percentage
        self.antialiasing = antialiasing
        self.isotropic = isotropic
        self._rand_zoom = None
        self.randomise()

    def randomise(self, spatial_rank=3):
        spatial_rank = int(cp.floor(spatial_rank))
        if self.isotropic:
            one_rand_zoom = cp.random.uniform(low=self._min_percentage,
                                              high=self._max_percentage)
            rand_zoom = cp.repeat(one_rand_zoom, spatial_rank)
        else:
            rand_zoom = cp.random.uniform(low=self._min_percentage,
                                          high=self._max_percentage,
                                          size=(spatial_rank,))
        self._rand_zoom = (rand_zoom + 100.0) / 100.0


    def _get_sigma(self, zoom):
        """
        Compute optimal standard deviation for Gaussian kernel.

            Cardoso et al., "Scale factor point spread function matching:
            beyond aliasing in image resampling", MICCAI 2015
        """
        k = 1 / zoom
        variance = (k ** 2 - 1 ** 2) * (2 * cp.sqrt(2 * cp.log(2))) ** (-2)
        sigma = cp.sqrt(variance)
        return sigma

    def _apply_transformation(self, image, interp_order=3):
        if interp_order < 0:
            return image
        assert self._rand_zoom is not None
        full_zoom = cp.array(self._rand_zoom)
        while len(full_zoom) < image.ndim:
            full_zoom = cp.hstack((full_zoom, [1.0]))
        is_undersampling = all(full_zoom[:3] < 1)
        run_antialiasing_filter = self.antialiasing and is_undersampling
        if run_antialiasing_filter:
            sigma = self._get_sigma(full_zoom[:3])
        if image.ndim == 4:
            output = []
            for mod in range(image.shape[-1]):
                to_scale = cpndi.gaussian_filter(image[..., mod], sigma) if \
                    run_antialiasing_filter else image[..., mod]
                scaled = cpndi.zoom(to_scale, full_zoom[:3], order=interp_order)
                output.append(scaled[:, :, :, cp.newaxis])
            return cp.concatenate(output, axis=-1)
        elif image.ndim == 3:
            to_scale = cpndi.gaussian_filter(image, sigma) \
                if run_antialiasing_filter else image
            scaled = cpndi.zoom(
                to_scale, full_zoom[:3], order=interp_order)
            return scaled[..., cp.newaxis]
        else:
            raise NotImplementedError('not implemented random scaling')

    def layer_op(self, inputs, isLabel = False):
        if inputs is None:
            return inputs
        if isLabel:
            order = 0
        else:
            order = 3
        result = []
        for i in inputs:
            result.append(self._apply_transformation(i, interp_order = order))
        return result
