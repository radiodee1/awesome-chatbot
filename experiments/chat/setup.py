from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
  'tensor2tensor'
]

setup(
    name='chat',
    version='0.1',
    author = 'D Liebman',
    author_email = 'david.c.liebman@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Chat Line Problem',
    requires=[]
)