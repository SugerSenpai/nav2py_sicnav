#!/usr/bin/env python

from setuptools import find_packages, setup

PROJECT = 'nav2py_sicnav_controller'

setup(name=PROJECT,
      version='1.0',
      description='nav2py_sicnav_controller',
      author='Volodymyr Shcherbyna',
      author_email='dev@voshch.dev',
      packages=find_packages(include=[PROJECT, PROJECT + '.*'])
      )
