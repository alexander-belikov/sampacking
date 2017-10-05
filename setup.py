import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="sampacking",
    version="0.1",
    author="Alexander Belikov",
    author_email="abelikov@gmail.com",
    description="tools to wrangle data",
    keywords="sampling bin packing",
    url="git@github.com:alexander-belikov/datahelpers.git",
    packages=['sampacking'],
    long_description=read('README'),
    install_requires=[]
    # include_dirs=[np.get_include()]
)
