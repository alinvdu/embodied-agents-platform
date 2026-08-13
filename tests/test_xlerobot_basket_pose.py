import unittest
from types import SimpleNamespace

from scripts.test_xlerobot_basket_pose import (
    _disable_right_gripper_torque,
    _enable_right_gripper_at_current_position,
    _without_gripper,
)


class _FakeBus:
    def __init__(self) -> None:
        self.calls = []

    def disable_torque(self, motor, *, num_retry):
        self.calls.append(("disable", motor, num_retry))

    def read(self, register, motor, *, num_retry):
        self.calls.append(("read", register, motor, num_retry))
        return 37.5

    def write(self, register, motor, value, *, num_retry):
        self.calls.append(("write", register, motor, value, num_retry))

    def enable_torque(self, motor, *, num_retry):
        self.calls.append(("enable", motor, num_retry))


class XLeRobotBasketPoseTests(unittest.TestCase):
    def test_gripper_is_reenabled_at_observed_position(self) -> None:
        bus = _FakeBus()
        robot = SimpleNamespace(bus2=bus)

        _disable_right_gripper_torque(robot)
        _enable_right_gripper_at_current_position(robot)

        self.assertEqual(
            bus.calls,
            [
                ("disable", "right_arm_gripper", 5),
                ("read", "Present_Position", "right_arm_gripper", 5),
                ("write", "Goal_Position", "right_arm_gripper", 37.5, 5),
                ("enable", "right_arm_gripper", 5),
            ],
        )

    def test_return_targets_do_not_move_gripper(self) -> None:
        targets = {
            "right_arm_shoulder_lift.pos": -44.0,
            "right_arm_gripper.pos": 20.0,
        }

        self.assertEqual(
            _without_gripper(targets),
            {"right_arm_shoulder_lift.pos": -44.0},
        )


if __name__ == "__main__":
    unittest.main()
