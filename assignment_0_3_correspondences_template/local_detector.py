import numpy as np
import math
import torch
import torch.nn.functional as F
import typing
from imagefiltering import * 

def gaussian_kernel1d(sigma: float) -> torch.Tensor:
    r"""Creates a 1D Gaussian kernel.
    Args:
        sigma (float): the standard deviation of the Gaussian distribution.
    Returns:
        torch.Tensor: the 1D Gaussian kernel.
    """
    ksize = get_gausskernel_size(sigma)
    x = torch.arange(ksize) - ksize // 2
    kernel = gaussian1d(x, sigma)
    return kernel

def guasian_deriv_kernel1d(sigma: float) -> torch.Tensor:
    r"""Creates a 1D Gaussian derivative kernel.
    Args:
        sigma (float): the standard deviation of the Gaussian distribution.
    Returns:
        torch.Tensor: the 1D Gaussian derivative kernel.
    """
    ksize = get_gausskernel_size(sigma)
    x = torch.arange(ksize) - ksize // 2
    kernel = gaussian_deriv1d(x, sigma)
    return kernel

def harris_response(x: torch.Tensor,
                     sigma_d: float,
                     sigma_i: float,
                     alpha: float = 0.04)-> torch.Tensor:
    r"""Computes the Harris cornerness function.The response map is computed according the following formulation:

    .. math::
        R = det(M) - alpha \cdot trace(M)^2

    where:

    .. math::
        M = \sum_{(x,y) \in W}
        \begin{bmatrix}
            I^{2}_x & I_x I_y \\
            I_x I_y & I^{2}_y \\
        \end{bmatrix}

    and :math:`k` is an empirically determined constant
    :math:`k ∈ [ 0.04 , 0.06 ]`

    Args:
        x: torch.Tensor: 4d tensor
        sigma_d (float): sigma of Gaussian derivative
        sigma_i (float): sigma of Gaussian blur, aka integration scale
        alpha (float): constant

    Return:
        torch.Tensor: Harris response

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`
    """
    I = spatial_gradient_first_order(x, sigma_d)

    I_x = I[:, :, 0, :, :]
    I_y = I[:, :, 1, :, :]

    I_x2 = gaussian_filter2d(I_x ** 2, sigma_i)
    I_y2 = gaussian_filter2d(I_y ** 2, sigma_i)
    I_xy = gaussian_filter2d(I_x * I_y, sigma_i)

    det = I_x2 * I_y2 - I_xy ** 2
    trace = I_x2 + I_y2

    R = det - alpha * trace ** 2

    return R


def nms2d(x: torch.Tensor, th: float = 0):
    r"""Applies non maxima suppression to the feature map in 3x3 neighborhood.
    Args:
        x: torch.Tensor: 4d tensor
        th (float): threshold
    Return:
        torch.Tensor: nmsed input

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, H, W)`
    """
    
    b, c, h, w = x.size()

    x_flat = x.view(b, c, -1)
    sorted_indices = torch.argsort(x_flat, dim=2, descending=True)
    result_indices = sorted_indices[:, :, 0]

    while sorted_indices.size(2) > 1:
        sorted_indices = sorted_indices[:, :, 1:]
        max_indices = sorted_indices[:, :, 0]
        
        
    
    out = torch.zeros_like(x)

    return out


def harris(x: torch.Tensor, sigma_d: float, sigma_i: float, th: float = 0):
    r"""Returns the coordinates of maximum of the Harris function.
    Args:
        x: torch.Tensor: 4d tensor
        sigma_d (float): scale
        sigma_i (float): scale
        th (float): threshold

    Return:
        torch.Tensor: coordinates of local maxima in format (b,c,h,w)

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(N, 4)`, where N - total number of maxima and 4 is (b,c,h,w) coordinates
    """
    # To get coordinates of the responces, you can use torch.nonzero function
    out = torch.zeros(0,2)
    return out


def create_scalespace(x: torch.Tensor, n_levels: int, sigma_step: float):
    r"""Creates an scale pyramid of image, usually used for local feature
    detection. Images are consequently smoothed with Gaussian blur.
    Args:
        x: torch.Tensor :math:`(B, C, H, W)`
        n_levels (int): number of the levels.
        sigma_step (float): blur step.

    Returns:
        Tuple(torch.Tensor, List(float)):
        1st output: image pyramid, (B, C, n_levels, H, W)
        2nd output: sigmas (coefficients for scale conversion)
    """

    b, ch, h, w = x.size()
    out = torch.zeros(b, ch, n_levels, h, w), [1.0 for x in range(n_levels)]
    return out


def nms3d(x: torch.Tensor, th: float = 0):
    r"""Applies non maxima suppression to the scale space feature map in 3x3x3 neighborhood.
    Args:
        x: torch.Tensor: 5d tensor
        th (float): threshold
    Shape:
      - Input: :math:`(B, C, D, H, W)`
      - Output: :math:`(B, C, D, H, W)`
    """
    out = torch.zeros_like(x)
    return out



def scalespace_harris_response(x: torch.Tensor,
                                n_levels: int = 40,
                                sigma_step: float = 1.1):
    r"""First computes scale space and then computes the Harris cornerness function 
    Args:
        x: torch.Tensor: 4d tensor
        n_levels (int): number of the levels, (default 40)
        sigma_step (float): blur step, (default 1.1)

    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(B, C, N_LEVELS, H, W)`, List(floats)
    """
    out = torch.zeros_like(x)
    return out



def scalespace_harris(x: torch.Tensor,
                       th: float = 0,
                       n_levels: int = 40,
                       sigma_step: float = 1.1):
    r"""Returns the coordinates of maximum of the Harris function.
    Args:
        x: torch.Tensor: 4d tensor
        th (float): threshold
        n_levels (int): number of scale space levels (default 40)
        sigma_step (float): blur step, (default 1.1)
        
    Shape:
      - Input: :math:`(B, C, H, W)`
      - Output: :math:`(N, 5)`, where N - total number of maxima and 5 is (b,c,d,h,w) coordinates
    """
    # To get coordinates of the responces, you can use torch.nonzero function
    # Don't forget to convert scale index to scale value with use of sigma
    out = torch.zeros(0,3)
    return out
