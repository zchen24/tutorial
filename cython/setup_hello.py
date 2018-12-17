from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='Hello World App',
    ext_modules=cythonize("hello.pyx")
)