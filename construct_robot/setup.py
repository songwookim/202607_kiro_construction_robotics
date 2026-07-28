from glob import glob

from setuptools import find_packages, setup

package_name = 'construct_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='irs',
    maintainer_email='thddn191@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cartesian_path_server = construct_robot.cartesian_path_server:main',
            'cartesian_path_client = construct_robot.cartesian_path_client:main',
        ],
    },
)
