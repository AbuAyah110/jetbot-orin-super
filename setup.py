import glob
import subprocess
from setuptools import setup, find_packages


def build_libs():
    try:
        subprocess.call(['cmake', '.'])
        subprocess.call(['make'])
    except Exception as exc:
        print('Warning: native library build skipped ({0})'.format(exc))


build_libs()


setup(
    name='jetbot-orin',
    version='0.5.0',
    description='JetBot for NVIDIA Jetson Orin Nano Super — educational AI robot platform',
    packages=find_packages(),
    install_requires=[
        'Adafruit_MotorHat',
        'Adafruit-SSD1306',
        'sparkfun-qwiic',
        'traitlets',
    ],
    package_data={'jetbot': ['ssd_tensorrt/*.so']},
)
