"""
Kernel implementations for Bayesian Optimization.

This module provides a unified interface for creating different kernel types
with appropriate hyperparameters for molecular descriptor optimization.
"""

from typing import Literal

from gpytorch.kernels import MaternKernel, RBFKernel, RQKernel, ScaleKernel


def get_kernel(
    kernel_name: Literal["matern", "rbf", "rq"],
    input_dim: int,
    ard: bool = True,
    **kwargs,
) -> ScaleKernel:
    """
    Create a kernel with automatic relevance determination (ARD).

    Args:
        kernel_name: Type of kernel ("matern", "rbf", or "rq")
        input_dim: Number of input dimensions
        ard: Whether to use ARD (recommended for high-dimensional inputs)
        **kwargs: Additional kernel-specific parameters

    Returns:
        ScaleKernel wrapping the base kernel

    Raises:
        ValueError: If kernel_name is not recognized
    """
    ard_num_dims = input_dim if ard else None

    kernel_name = kernel_name.lower()

    if kernel_name == "matern":
        # Matérn kernel with ν=2.5 (good balance of smoothness and flexibility)
        base_kernel = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
    elif kernel_name == "rbf":
        # RBF (Gaussian) kernel (infinitely smooth)
        base_kernel = RBFKernel(ard_num_dims=ard_num_dims)
    elif kernel_name == "rq":
        # Rational Quadratic kernel (mixture of RBF kernels with different lengthscales)
        # Extract alpha parameter if provided, otherwise use default
        alpha = kwargs.get("alpha", 1.0)
        base_kernel = RQKernel(alpha=alpha, ard_num_dims=ard_num_dims)
    else:
        raise ValueError(
            f"Unknown kernel '{kernel_name}'. " "Choose from: 'matern', 'rbf', 'rq'"
        )

    # Wrap with ScaleKernel to learn output scale
    return ScaleKernel(base_kernel)
