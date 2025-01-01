from setuptools import setup

setup(name='hockey',
      version='2.6',
      description='Simple Hockey Environments',
      url='https://github.com/martius-lab',
      author='Georg Martius, Uni Tuebingen, Autonomous Learning',
      author_email='georg.martius@uni-tuebingen.de',
      license='MIT',
      packages=['hockey'],
      python_requires='>=3.6',
      install_requires=['gymnasium', 'numpy', 'box2d-py','pygame'],
      zip_safe=False)
