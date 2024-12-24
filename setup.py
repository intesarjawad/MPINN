from setuptools import setup, find_packages

setup(
    name="mpinn",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'tensorflow>=2.6.0',
        'numpy>=1.19.2',
        'matplotlib>=3.4.0',
        'pyyaml>=5.4.0'
    ]
) 