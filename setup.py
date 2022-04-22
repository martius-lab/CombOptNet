#!/usr/bin/env python
from distutils.core import setup

setup(
    name="constraint_handler",
    version="1.0",
    description="Handler for initializing a learnable set of constraints.",
    author="Anselm Paulus",
    author_email="anselm.paulus@tuebingen.mpg.de",
    url=None,
    packages=["constraint_handler"],
    install_requires=[
        "torch>=1.4.0",
        "numpy"
    ],
)