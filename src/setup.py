"""
setup for stldeli
find all the packages in src
"""
from setuptools import setup, find_packages

setup(
    name='stldeli',
    version='0.0.0',
    packages=find_packages(),
    package_dir={'': '.'},
    url='',
    license='MIT',
    author='Tom Wilson',
    author_email='tom.andrew.wilson@gmail.com',
    description='slice arbitrary stl file into optimal gcode'
)
