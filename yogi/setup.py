from setuptools import setup

setup(name='yogi',
      version='0.0.1',
      install_requires=['alembic',
                        'click',
                        'columnar',
                        'flask',
                        'flask-sqlalchemy',
                        'imageio',
                        'imageio-ffmpeg',
                        'scikit-image',
                        'sqlalchemy',],
      scripts=['bin/yogi'],
)
