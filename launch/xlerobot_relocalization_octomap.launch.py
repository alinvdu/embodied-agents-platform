from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, ThisLaunchFileDir
from launch_ros.actions import Node


def generate_launch_description():
    cloud_topic = LaunchConfiguration("cloud_topic")
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "cloud_topic",
                default_value="/camera/head/points",
                description="PointCloud2 topic to insert into the temporary relocalization OctoMap.",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [ThisLaunchFileDir(), "..", "config", "xlerobot_relocalization_octomap.yaml"]
                ),
                description="Temporary relocalization OctoMap parameter file.",
            ),
            Node(
                package="octomap_server",
                executable="octomap_server_node",
                name="relocalization_octomap_server",
                output="screen",
                parameters=[params_file],
                remappings=[
                    ("cloud_in", cloud_topic),
                    ("projected_map", "/relocalization_projected_map"),
                    ("projected_map_updates", "/relocalization_projected_map_updates"),
                    ("octomap_binary", "/relocalization_octomap_binary"),
                    ("octomap_full", "/relocalization_octomap_full"),
                ],
            ),
        ]
    )
