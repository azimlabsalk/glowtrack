from setuptools import setup, find_packages

setup(
    name="wink",
    version="0.1",
    package_dir={'': 'src/python'},
    packages=find_packages(),
    install_requires=['click',
                      'imageio',
                      'opencv-contrib-python==3.4.2.17',
                      'opencv-python==4.0.0.21',
                      'pyserial']
)

