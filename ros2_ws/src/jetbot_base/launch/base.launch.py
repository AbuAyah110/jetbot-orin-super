from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('jetbot_base'),
        'config',
        'base.yaml',
    )
    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=default_config),
        DeclareLaunchArgument('backend', default_value='mock'),
        Node(
            package='jetbot_base',
            executable='base_node',
            name='jetbot_base',
            output='screen',
            parameters=[{
                'config_file': LaunchConfiguration('config_file'),
                'backend': LaunchConfiguration('backend'),
            }],
        ),
    ])
