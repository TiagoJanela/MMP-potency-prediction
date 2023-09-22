from setuptools import setup

INSTALL_REQUIRES = [
    'numpy>=1.21',
    'matplotlib>=3.1',
    'typing_extensions; python_version < "3.8"',
    'scipy',
    'scikit-learn'
]

setup(
    name='sveta',
    version='0.1',
    packages=['sveta'],
    url='',
    license='MIT',
    author='cfeldmann',
    author_email='christian.w.feldmann@gmail.com',
    description='A package to calculate Shapley values for support vector based machine learning models.',
    install_requires=INSTALL_REQUIRES
    )
