# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:25:51 2021

@author: ariken
"""
from __future__ import absolute_import, print_function

import cupy as cp
import cupyx.scipy.ndimage as cpndi

class RandomRotationLayer(object):
    """
    generate randomised rotation matrix for data augmentation
    """

    def __init__(self, name='random_rotation', uniform_angle = True):
        self._transform = None
        self.min_angle = None
        self.max_angle = None
        self.rotation_angle_x = None
        self.rotation_angle_y = None
        self.rotation_angle_z = None
        if uniform_angle:
            self.init_uniform_angle()
        else:
            self.init_non_uniform_angle()
        self.randomise()
        
        
    def init_uniform_angle(self, rotation_angle=(-10.0, 10.0)):
        assert rotation_angle[0] < rotation_angle[1]
        self.min_angle = float(rotation_angle[0])
        self.max_angle = float(rotation_angle[1])


    def init_non_uniform_angle(self,
                               rotation_angle_x,
                               rotation_angle_y,
                               rotation_angle_z):
        if len(rotation_angle_x):
            assert rotation_angle_x[0] < rotation_angle_x[1]
        if len(rotation_angle_y):
            assert rotation_angle_y[0] < rotation_angle_y[1]
        if len(rotation_angle_z):
            assert rotation_angle_z[0] < rotation_angle_z[1]
        self.rotation_angle_x = [float(e) for e in rotation_angle_x]
        self.rotation_angle_y = [float(e) for e in rotation_angle_y]
        self.rotation_angle_z = [float(e) for e in rotation_angle_z]


    def randomise(self, spatial_rank=3):
        if spatial_rank == 3:
            self._randomise_transformation_3d()
        else:
            # currently not supported spatial rank for rand rotation
            pass


    def _randomise_transformation_3d(self):
        angle_x = 0.0
        angle_y = 0.0
        angle_z = 0.0
        if self.min_angle is None and self.max_angle is None:
            # generate transformation
            if len(self.rotation_angle_x) >= 2:
                angle_x = cp.random.uniform(
                    self.rotation_angle_x[0],
                    self.rotation_angle_x[1]) * cp.pi / 180.0

            if len(self.rotation_angle_y) >= 2:
                angle_y = cp.random.uniform(
                    self.rotation_angle_y[0],
                    self.rotation_angle_y[1]) * cp.pi / 180.0

            if len(self.rotation_angle_z) >= 2:
                angle_z = cp.random.uniform(
                    self.rotation_angle_z[0],
                    self.rotation_angle_z[1]) * cp.pi / 180.0
        else:
            # generate transformation
            angle_x = cp.random.uniform(
                self.min_angle, self.max_angle) * cp.pi / 180.0
            angle_y = cp.random.uniform(
                self.min_angle, self.max_angle) * cp.pi / 180.0
            angle_z = cp.random.uniform(
                self.min_angle, self.max_angle) * cp.pi / 180.0

        
        transform_x = cp.array([[float(cp.cos(angle_x)), float(-cp.sin(angle_x)), 0.0],
                                [float(cp.sin(angle_x)), float(cp.cos(angle_x)), 0.0],
                                [0.0, 0.0, 1.0]])
        transform_y = cp.array([[float(cp.cos(angle_y)), 0.0, float(cp.sin(angle_y))],
                                [0.0, 1.0, 0.0],
                                [float(-cp.sin(angle_y)), 0.0, float(cp.cos(angle_y))]])
        transform_z = cp.array([[1.0, 0.0, 0.0],
                                [0.0, float(cp.cos(angle_z)), float(-cp.sin(angle_z))],
                                [0.0, float(cp.sin(angle_z)), float(cp.cos(angle_z))]])
        transform = cp.dot(transform_z, cp.dot(transform_x, transform_y))
        self._transform = transform

    def _apply_transformation_3d(self, image_3d, interp_order=3):
        if interp_order < 0:
            return image_3d
        assert image_3d.ndim == 3
        assert self._transform is not None
        assert all([dim > 1 for dim in image_3d.shape]), \
            'random rotation supports 3D inputs only'
        center_ = 0.5 * cp.asarray(image_3d.shape, dtype=cp.int64)
        c_offset = center_ - center_.dot(self._transform)
        image_3d[...] = cpndi.affine_transform(
            image_3d[...], self._transform.T, c_offset, order=interp_order)
        return image_3d

    def layer_op(self, inputs, isLabel = False):
        if inputs is None:
            return inputs
        if isLabel:
            order = 0
        else:
            order = 3
        for i in inputs:
            for channel_idx in range(i.shape[-1]):
                if i.ndim == 4:
                    i[..., channel_idx] = \
                        self._apply_transformation_3d(
                            i[..., channel_idx], interp_order = order)
                else:
                    raise NotImplementedError("unknown input format")
                # shapes = []
                # for (field, image) in inputs.items():
                #     shapes.append(image.shape)
                # assert(len(shapes) == 2 and shapes[0][0:4] == shapes[1][0:4]), shapes
        return inputs