# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Tuple

import torch
import torch.nn as nn

from ...structures.meshes import Meshes

# A renderer class should be initialized with a
# function for rasterization and a function for shading.
# The rasterizer should:
#     - transform inputs from world -> screen space
#     - rasterize inputs
#     - return fragments
# The shader can take fragments as input along with any other properties of
# the scene and generate images.

# E.g. rasterize inputs and then shade
#
# fragments = self.rasterize(meshes)
# images = self.shader(fragments, meshes)
# return images


class MeshRenderer(nn.Module):
    """
    一个用于渲染异构网格批次的类。该类应通过光栅化器（MeshRasterizer或MeshRasterizerOpenGL）和着色器类进行初始化，这两个组件各自拥有前向处理功能。
    """

    def __init__(self, rasterizer, shader) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.shader.to(device)
        return self

    def forward(self, meshes_world: Meshes, **kwargs) -> torch.Tensor:
        """
        通过光栅化和着色处理，从一组网格渲染出一批图像。

        注意：如果光栅化的模糊半径大于0.0，某些像素可能会有一个或多个重心坐标超出[0, 1]范围。
        对于相对于面f存在越界重心坐标的像素，在插值纹理uv坐标和z缓冲区之前需要进行裁剪，
        以确保颜色和深度限制在该面对应的范围内。
        为此，请设置rasterizer.raster_settings.clip_barycentric_coords=True
        """
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        return images


class MeshRendererWithFragments(nn.Module):
    """
    A class for rendering a batch of heterogeneous meshes. The class should
    be initialized with a rasterizer (a MeshRasterizer or a MeshRasterizerOpenGL)
    and shader class which each have a forward function.

    In the forward pass this class returns the `fragments` from which intermediate
    values such as the depth map can be easily extracted e.g.

    .. code-block:: python
        images, fragments = renderer(meshes)
        depth = fragments.zbuf
    """

    def __init__(self, rasterizer, shader) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.shader.to(device)
        return self

    def forward(
        self, meshes_world: Meshes, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render a batch of images from a batch of meshes by rasterizing and then
        shading.

        NOTE: If the blur radius for rasterization is > 0.0, some pixels can
        have one or more barycentric coordinates lying outside the range [0, 1].
        For a pixel with out of bounds barycentric coordinates with respect to a
        face f, clipping is required before interpolating the texture uv
        coordinates and z buffer so that the colors and depths are limited to
        the range for the corresponding face.
        For this set rasterizer.raster_settings.clip_barycentric_coords=True
        """
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        return images, fragments
