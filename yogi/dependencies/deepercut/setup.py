from setuptools import setup

setup(name='deepercut',
      version='0.0.1',
      install_requires=['numpy',
                        'scikit-image',
                        'pillow',
                        'scipy==1.2.1',
                        'pyyaml',
                        'matplotlib',
                        'cython',
                        'tensorflow-gpu==1.8.0',
                        'easydict',
                        'munkres',
                        ],
)
