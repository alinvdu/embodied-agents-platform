from __future__ import annotations

import unittest

from examples.robot42_agent_backend import _merge_args, build_parser
from xlerobot_agent.home_agent import HomeAgentConfig


class Robot42AgentBackendTests(unittest.TestCase):
    def test_no_navigation_waypoint_breakdown_flag_disables_segmentation(self) -> None:
        args = build_parser().parse_args(["--no-navigation-waypoint-breakdown"])

        config = _merge_args(HomeAgentConfig(), args)

        self.assertFalse(config.navigation_waypoint_breakdown_enabled)

    def test_navigation_waypoint_breakdown_default_preserves_config(self) -> None:
        args = build_parser().parse_args([])

        config = _merge_args(HomeAgentConfig(navigation_waypoint_breakdown_enabled=False), args)

        self.assertFalse(config.navigation_waypoint_breakdown_enabled)


if __name__ == "__main__":
    unittest.main()
