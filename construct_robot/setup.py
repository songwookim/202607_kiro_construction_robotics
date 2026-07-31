from glob import glob

from setuptools import find_packages, setup

package_name = 'construct_robot'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='irs',
    maintainer_email='thddn191@gmail.com',
    description='KIRO hardware launch, Cartesian action, and weld GUI nodes',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cartesian_path_server = construct_robot.cartesian_path_server:main',
            'cartesian_path_client = construct_robot.cartesian_path_client:main',
            'weld_action_gui = construct_robot.weld_action_gui:main',
            'viser_viewer = construct_robot.viser_viewer:main',
            'h600_modbus_bridge = construct_robot.h600_modbus_bridge:main',
            'h600_modbus_gui = construct_robot.h600_modbus_gui:main',
            'weld_stack_supervisor = construct_robot.weld_stack_supervisor:main',
            'rviz_goal_state_sync = construct_robot.rviz_goal_state_sync:main',
        ],
    },
)
