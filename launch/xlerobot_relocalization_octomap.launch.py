# Copyright 2026 Alin Vasile Dumitru
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
