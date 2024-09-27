from setuptools import find_packages, setup

setup(
    name="rl_baselines",
    version="0.0.1",  # the current version of your package
    packages=find_packages(),  # automatically discover all packages and subpackages
    url="https://github.com/erickTornero/rl-baselines",  # replace with the URL of your project
    author="Erick Tornero",
    author_email="erickdeivy01@gmail.com",
    description="Implementation of Reinforcement Learning Algorithms",  # replace with a brief description of your project
    install_requires=[
        # list of packages your project depends on
        # you can specify versions as well, e.g. 'numpy>=1.15.1'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)