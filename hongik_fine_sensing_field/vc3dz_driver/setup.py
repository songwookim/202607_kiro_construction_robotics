from setuptools import setup

package_name = 'vc3dz_driver'

setup(
    name=package_name,
    version='0.1.0',
    description='Pure-Python TCP driver for the VC nano 3D-Z laser line profiler (port 1096), ROS-independent',
    author='DBCM Lab',
    license='Apache-2.0',
    packages=[package_name],
    python_requires='>=3.10',
    install_requires=['numpy>=1.21,<2'],      # numpy 2 untested on the ROS Humble target (tools/python_env.yaml)
    extras_require={'h5': ['h5py'], 'test': ['pytest<8']},
    # ament_python 용 (colcon build). pip 설치에서는 무해.
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    entry_points={
        'console_scripts': [
            'vc3dz = vc3dz_driver.cli:main',
            'vc3dz_replay = vc3dz_driver.replay:main',
            'vc3dz_fake_sensor = vc3dz_driver.fake_sensor:main',
        ],
    },
    zip_safe=False,
)
