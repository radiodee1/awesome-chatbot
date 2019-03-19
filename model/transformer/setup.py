from . import problem

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
  'tensor2tensor'
]

setup(
    name='transformer',
    version='0.1',
    author = 'David Liebman',
    author_email = '',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Chat Line Problem',
    requires=[]
)