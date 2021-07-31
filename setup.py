from setuptools import setup, find_packages
import os

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'

install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(
    name='drltrader',
    url='https://github.com/sleung852/drltrader',
    license='MIT',
    author='See Ho Leung',
    author_email='seeleung@connect.hku.hk',
    description='A deep reinforcement learning library for training algo trading bots',
    packages=find_packages(exclude=['test']),
    install_requires=install_requires,
)
                             