from setuptools import setup, find_packages

requirements = [
    'torch',
    'scikit-learn',
    'numpy',
    'matplotlib',
    'scipy',
    'pyyaml'
]

VERSION = '0.0.1'

setup(
    name='GlasXC',
    version=VERSION,
    url='https://github.com/pyschedelicsid/GlasXC',
    description='Python module for XMC with Glas Regularizer',

    packages=find_packages(),

    zip_safe=True,
    install_requires=requirements
)
