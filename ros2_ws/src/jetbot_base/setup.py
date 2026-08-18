from setuptools import setup
import os
from glob import glob

package_name = 'jetbot_base'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools', 'PyYAML'],
    zip_safe=True,
    maintainer='jetbot-orin-super',
    maintainer_email='dev@localhost',
    description='Deterministic JetBot base node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'base_node = jetbot_base.base_node:main',
            'teleop_keyboard = jetbot_base.teleop_keyboard:main',
        ],
    },
)
