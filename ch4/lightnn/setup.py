#!/usr/bin/env python
# coding=utf-8

from distutils.core import setup
from setuptools import find_packages

  
PACKAGE = "lightnn"  
NAME = "lightnn"  
DESCRIPTION = "This package can download funds data from https://github.com/l11x0m7/lightnn. For details, please visit https://skyhigh233.com."  
AUTHOR = "Xuming Lin"  
AUTHOR_EMAIL = "l11x0m7@gmail.com"  
URL = "https://github.com/l11x0m7/lightnn"  
VERSION = __import__(PACKAGE).__version__  
    
setup(  
        name=NAME,  
        version=VERSION,  
        description=DESCRIPTION,  
        # long_description=open("README.md", 'rb').readlines(),  
        author=AUTHOR,  
        author_email=AUTHOR_EMAIL,  
        license="Apache License, Version 2.0",  
        url=URL,  
        packages=find_packages('./'),  
        classifiers=[  
                    "Development Status :: 3 - Alpha",  
                    "Intended Audience :: Developers",  
                    "Operating System :: OS Independent",  
                    "Programming Language :: Python",  
                ],  
        zip_safe=False,  
    )  
