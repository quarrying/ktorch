import sys
from setuptools import find_packages, setup

install_requires = ['numpy>=1.11.1', 'opencv-python', 'pillow']

setup(
    name='ktorch',
    version='0.1.0',
    description='Some extensions for PyTorch and utilities using Pytorch',
    long_description='Some extensions for PyTorch and utilities using Pytorch',
    keywords='machine learning',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Utilities',
    ],
    url='',
    author='quarryman',
    author_email='quarrying@qq.com',
    license='GPLv3',
    install_requires=install_requires,
    zip_safe=False)
