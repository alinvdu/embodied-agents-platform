from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cloud_topic = LaunchConfiguration("cloud_topic")
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "cloud_topic",
                default_value="/camera/head/points",
                description="PointCloud2 topic to insert into OctoMap.",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value="/home/alin/Robot42/config/xlerobot_octomap.yaml",
                description="OctoMap server parameter file.",
            ),
            Node(
                package="octomap_server",
                executable="octomap_server_node",
                name="octomap_server",
                output="screen",
                parameters=[params_file],
                remappings=[
                    ("cloud_in", cloud_topic),
                ],
            ),
        ]
    )
