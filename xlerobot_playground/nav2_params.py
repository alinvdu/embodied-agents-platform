from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def rectangular_footprint(
    *,
    length_m: float,
    width_m: float,
    center_x_m: float = 0.0,
    center_y_m: float = 0.0,
) -> str:
    half_length = max(float(length_m), 0.0) / 2.0
    half_width = max(float(width_m), 0.0) / 2.0
    points = [
        (round(center_x_m + half_length, 4), round(center_y_m + half_width, 4)),
        (round(center_x_m + half_length, 4), round(center_y_m - half_width, 4)),
        (round(center_x_m - half_length, 4), round(center_y_m - half_width, 4)),
        (round(center_x_m - half_length, 4), round(center_y_m + half_width, 4)),
    ]
    return "[" + ", ".join(f"[{x}, {y}]" for x, y in points) + "]"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping at {path}, got {type(data).__name__}")
    return data


def dump_yaml(path: str | Path, data: dict[str, Any]) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _remove_critic(root: dict[str, Any], critic_name: str) -> None:
    critics = root.get("critics")
    if isinstance(critics, list):
        root["critics"] = [item for item in critics if item != critic_name]
    prefix = f"{critic_name}."
    for key in list(root):
        if key.startswith(prefix):
            root.pop(key, None)


def render_slam_toolbox_params(
    *,
    use_sim_time: bool = True,
    scan_topic: str = "/scan",
    map_frame: str = "map",
    odom_frame: str = "odom",
    base_frame: str = "base_link",
    resolution: float = 0.05,
    max_laser_range: float = 10.0,
) -> dict[str, Any]:
    return {
        "slam_toolbox": {
            "ros__parameters": {
                "use_sim_time": use_sim_time,
                "slam_mode": "mapping",
                "mode": "mapping",
                "map_frame": map_frame,
                "odom_frame": odom_frame,
                "base_frame": base_frame,
                "scan_topic": scan_topic,
                "transform_publish_period": 0.05,
                "map_update_interval": 2.0,
                "resolution": resolution,
                "max_laser_range": max_laser_range,
                "minimum_time_interval": 0.1,
                "throttle_scans": 1,
                "queue_size": 50,
                "enable_interactive_mode": True,
                "debug_logging": False,
            }
        }
    }


def patch_nav2_params(
    base_params: dict[str, Any],
    *,
    use_sim_time: bool = True,
    map_frame: str = "map",
    odom_frame: str = "odom",
    base_frame: str = "base_link",
    scan_topic: str = "/scan",
    global_map_topic: str = "/projected_map",
    robot_radius: float = 0.24,
    footprint: str | None = None,
    footprint_padding: float = 0.0,
    obstacle_max_range: float = 9.5,
    raytrace_max_range: float = 10.0,
    inflation_radius: float = 0.0,
    local_observation_persistence_s: float = 0.35,
    local_costmap_width: int = 4,
    local_costmap_height: int = 4,
    voxel_origin_z: float = 0.0,
    voxel_z_resolution: float = 0.05,
    voxel_z_voxels: int = 32,
    max_linear_velocity: float = 0.65,
    max_angular_velocity: float = 0.45,
    min_linear_velocity_threshold: float = 0.01,
    min_angular_velocity_threshold: float = 0.02,
    min_speed_theta: float = 0.02,
    trans_stopped_velocity: float = 0.01,
    follow_path_xy_goal_tolerance_m: float = 0.10,
    path_align_scale: float = 4.0,
    goal_align_scale: float = 0.0,
    oscillation_reset_dist_m: float = 0.01,
    oscillation_reset_angle_rad: float = 0.05,
    oscillation_reset_time_s: float = 5.0,
    goal_dist_scale: float = 8.0,
    rotate_to_goal_scale: float = 0.0,
    rotate_to_goal_slowing_factor: float = 1.0,
    transform_tolerance_s: float = 0.5,
    progress_required_movement_radius: float = 0.01,
    progress_movement_time_allowance_s: float = 60.0,
    xy_goal_tolerance_m: float = 0.18,
    yaw_goal_tolerance_rad: float = 3.14,
    enable_local_scan_obstacles: bool = False,
) -> dict[str, Any]:
    params = deepcopy(base_params)

    def set_all_use_sim_time(value: bool) -> None:
        def visit(item: Any) -> None:
            if isinstance(item, dict):
                ros_params = item.get("ros__parameters")
                if isinstance(ros_params, dict):
                    ros_params["use_sim_time"] = value
                for child in item.values():
                    visit(child)
            elif isinstance(item, list):
                for child in item:
                    visit(child)

        visit(params)

    def node_params(node_name: str) -> dict[str, Any]:
        node = params.setdefault(node_name, {})
        if isinstance(node, dict) and "ros__parameters" in node:
            return node.setdefault("ros__parameters", {})
        nested = node.get(node_name)
        if isinstance(nested, dict):
            return nested.setdefault("ros__parameters", {})
        return node.setdefault("ros__parameters", {})

    def set_if_present(mapping: dict[str, Any], key: str, value: Any) -> None:
        mapping[key] = value

    set_all_use_sim_time(use_sim_time)

    for node_name in (
        "amcl",
        "behavior_server",
        "bt_navigator",
        "controller_server",
        "global_costmap",
        "local_costmap",
        "map_server",
        "planner_server",
        "smoother_server",
        "velocity_smoother",
        "waypoint_follower",
    ):
        ros_params = node_params(node_name)
        set_if_present(ros_params, "use_sim_time", use_sim_time)

    for costmap_name, global_frame in (
        ("global_costmap", map_frame),
        ("local_costmap", odom_frame),
    ):
        root = node_params(costmap_name)
        root["global_frame"] = global_frame
        root["robot_base_frame"] = base_frame
        root["use_sim_time"] = use_sim_time
        root["transform_tolerance"] = transform_tolerance_s
        if footprint:
            root["footprint"] = deepcopy(footprint)
            root.pop("robot_radius", None)
        else:
            root["robot_radius"] = robot_radius
            root.pop("footprint", None)
        root["footprint_padding"] = footprint_padding
        if costmap_name == "local_costmap":
            root["width"] = local_costmap_width
            root["height"] = local_costmap_height

        plugins = list(root.get("plugins", []))
        if costmap_name == "global_costmap":
            plugins = [plugin for plugin in plugins if plugin not in {"obstacle_layer", "voxel_layer"}]
            if "static_layer" not in plugins:
                plugins.insert(0, "static_layer")
            root["plugins"] = plugins
            root.pop("obstacle_layer", None)
            root.pop("voxel_layer", None)
            static_layer = root.setdefault("static_layer", {})
            static_layer["plugin"] = "nav2_costmap_2d::StaticLayer"
            static_layer["map_topic"] = global_map_topic
            static_layer["subscribe_to_updates"] = True
            static_layer["map_subscribe_transient_local"] = False
            static_layer["enabled"] = True
            static_layer["transform_tolerance"] = transform_tolerance_s
        else:
            if not enable_local_scan_obstacles:
                plugins = [plugin for plugin in plugins if plugin not in {"obstacle_layer", "voxel_layer"}]
                if "static_layer" not in plugins:
                    plugins.insert(0, "static_layer")
                root["plugins"] = plugins
                root.pop("obstacle_layer", None)
                root.pop("voxel_layer", None)
                static_layer = root.setdefault("static_layer", {})
                static_layer["plugin"] = "nav2_costmap_2d::StaticLayer"
                static_layer["map_topic"] = global_map_topic
                static_layer["subscribe_to_updates"] = True
                static_layer["map_subscribe_transient_local"] = False
                static_layer["enabled"] = True
                static_layer["transform_tolerance"] = transform_tolerance_s
        if inflation_radius <= 0.0 and "inflation_layer" in plugins:
            plugins = [plugin for plugin in plugins if plugin != "inflation_layer"]
            root["plugins"] = plugins
            root.pop("inflation_layer", None)
        if "obstacle_layer" in plugins:
            obstacle_layer = root.setdefault("obstacle_layer", {})
            obstacle_layer["observation_sources"] = "scan"
            obstacle_layer["scan"] = {
                "topic": scan_topic,
                "data_type": "LaserScan",
                "clearing": True,
                "marking": True,
                "inf_is_valid": True,
                "raytrace_max_range": raytrace_max_range,
                "obstacle_max_range": obstacle_max_range,
                "max_obstacle_height": 2.0,
                "observation_persistence": local_observation_persistence_s,
            }

        if "voxel_layer" in plugins:
            voxel_layer = root.setdefault("voxel_layer", {})
            voxel_layer["observation_sources"] = "scan"
            voxel_layer["origin_z"] = voxel_origin_z
            voxel_layer["z_resolution"] = voxel_z_resolution
            voxel_layer["z_voxels"] = voxel_z_voxels
            voxel_layer["max_obstacle_height"] = max(
                voxel_origin_z + voxel_z_resolution * voxel_z_voxels,
                2.0,
            )
            voxel_layer["scan"] = {
                "topic": scan_topic,
                "data_type": "LaserScan",
                "clearing": True,
                "marking": True,
                "inf_is_valid": True,
                "raytrace_max_range": raytrace_max_range,
                "obstacle_max_range": obstacle_max_range,
                "max_obstacle_height": 2.0,
                "observation_persistence": local_observation_persistence_s,
            }

        if inflation_radius > 0.0 and "inflation_layer" in plugins:
            inflation = root.setdefault("inflation_layer", {})
            inflation["inflation_radius"] = inflation_radius

    bt = node_params("bt_navigator")
    bt["global_frame"] = map_frame
    bt["robot_base_frame"] = base_frame
    bt["odom_topic"] = "/odom"

    behavior = node_params("behavior_server")
    behavior["global_frame"] = odom_frame
    behavior["robot_base_frame"] = base_frame
    behavior["transform_tolerance"] = transform_tolerance_s
    if "max_rotational_vel" in behavior:
        behavior["max_rotational_vel"] = max_angular_velocity
    if "min_rotational_vel" in behavior:
        behavior["min_rotational_vel"] = min(float(behavior["min_rotational_vel"]), max_angular_velocity)

    controller = node_params("controller_server")
    controller["odom_topic"] = "/odom"
    controller["transform_tolerance"] = transform_tolerance_s
    controller["min_x_velocity_threshold"] = min_linear_velocity_threshold
    controller["min_theta_velocity_threshold"] = min_angular_velocity_threshold
    progress_checker = controller.setdefault("progress_checker", {})
    progress_checker["plugin"] = progress_checker.get("plugin", "nav2_controller::SimpleProgressChecker")
    progress_checker["required_movement_radius"] = progress_required_movement_radius
    progress_checker["movement_time_allowance"] = progress_movement_time_allowance_s
    controller["goal_checker_plugins"] = ["goal_checker"]
    controller.pop("general_goal_checker", None)
    goal_checker = controller.setdefault("goal_checker", {})
    goal_checker["plugin"] = "nav2_controller::PositionGoalChecker"
    goal_checker["xy_goal_tolerance"] = xy_goal_tolerance_m
    goal_checker["stateful"] = True
    follow_path = controller.get("FollowPath")
    if isinstance(follow_path, dict):
        follow_path["max_vel_x"] = max_linear_velocity
        follow_path["max_speed_xy"] = max_linear_velocity
        follow_path["max_vel_theta"] = max_angular_velocity
        follow_path["min_speed_theta"] = min(min_speed_theta, max_angular_velocity)
        follow_path["trans_stopped_velocity"] = trans_stopped_velocity
        follow_path["xy_goal_tolerance"] = max(
            0.0,
            min(float(follow_path_xy_goal_tolerance_m), float(xy_goal_tolerance_m)),
        )
        if path_align_scale <= 0.0:
            _remove_critic(follow_path, "PathAlign")
        elif "PathAlign.scale" in follow_path:
            follow_path["PathAlign.scale"] = path_align_scale
            follow_path["PathAlign.forward_point_distance"] = min(
                float(follow_path.get("PathAlign.forward_point_distance", 0.1)),
                0.1,
            )
        if goal_align_scale <= 0.0:
            _remove_critic(follow_path, "GoalAlign")
        elif "GoalAlign.scale" in follow_path:
            follow_path["GoalAlign.scale"] = goal_align_scale
        if "Oscillation" in follow_path.get("critics", []):
            follow_path["Oscillation.reset_dist"] = oscillation_reset_dist_m
            follow_path["Oscillation.reset_angle"] = oscillation_reset_angle_rad
            follow_path["Oscillation.reset_time"] = oscillation_reset_time_s
        if "GoalDist.scale" in follow_path:
            follow_path["GoalDist.scale"] = goal_dist_scale
        if rotate_to_goal_scale <= 0.0:
            _remove_critic(follow_path, "RotateToGoal")
        elif "RotateToGoal.scale" in follow_path:
            follow_path["RotateToGoal.scale"] = rotate_to_goal_scale
        if "RotateToGoal.slowing_factor" in follow_path:
            follow_path["RotateToGoal.slowing_factor"] = rotate_to_goal_slowing_factor

    planner = node_params("planner_server")
    planner["expected_planner_frequency"] = planner.get("expected_planner_frequency", 5.0)

    velocity_smoother = node_params("velocity_smoother")
    if isinstance(velocity_smoother.get("max_velocity"), list) and len(velocity_smoother["max_velocity"]) >= 3:
        velocity_smoother["max_velocity"][0] = max_linear_velocity
        velocity_smoother["max_velocity"][2] = max_angular_velocity
    if isinstance(velocity_smoother.get("min_velocity"), list) and len(velocity_smoother["min_velocity"]) >= 3:
        velocity_smoother["min_velocity"][0] = -max_linear_velocity
        velocity_smoother["min_velocity"][2] = -max_angular_velocity
    if not isinstance(velocity_smoother.get("deadband_velocity"), list) or len(velocity_smoother["deadband_velocity"]) < 3:
        velocity_smoother["deadband_velocity"] = [0.0, 0.0, 0.0]
    velocity_smoother["deadband_velocity"][0] = min_linear_velocity_threshold
    velocity_smoother["deadband_velocity"][2] = min_angular_velocity_threshold

    amcl = node_params("amcl")
    if amcl:
        amcl["base_frame_id"] = base_frame
        amcl["global_frame_id"] = map_frame
        amcl["odom_frame_id"] = odom_frame
        amcl["scan_topic"] = scan_topic.lstrip("/")
        amcl["tf_broadcast"] = False

    return params
