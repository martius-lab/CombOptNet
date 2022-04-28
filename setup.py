#!/usr/bin/env python
from distutils.core import setup

setup(
    name="comboptnet_module",
    version="1.0",
    description="Differentiable ILP solver module from 'CombOptNet: Fit the Right NP-Hard Problem by Learning Integer Programming Constraints' (ICML 2021, Paulus et al.)",
    author="Anselm Paulus",
    author_email="anselm.paulus@tuebingen.mpg.de",
    url=None,
    packages=["comboptnet_module"],
    install_requires=[
        "torch>=1.4.0",
        "numpy>=1.16.0",
        "ray>=1.0.0",
        "jax>=0.2.12",
        "jaxlib>=0.1.64",
        "gurobipy>=9.1.0",
    ],
)