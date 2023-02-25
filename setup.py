from setuptools import setup, find_packages

setup(
    name='mujoco_arm',
    version='0.1',
    packages=find_packages(),
    author="Varad Vaidya",
    author_email="vaidyavarad2001@gmail.com",
    install_requires=['numpy','matplotlib','mujoco']
)
