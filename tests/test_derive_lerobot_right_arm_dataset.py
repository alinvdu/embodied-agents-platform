import unittest

from scripts.derive_lerobot_right_arm_dataset import (
    RIGHT_ARM_NAMES,
    _derive_global_stats,
    _derive_info,
    _feature_indices,
)


class DeriveRightArmDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.all_names = [f"left_{index}" for index in range(6)] + list(RIGHT_ARM_NAMES) + [
            "head_pan",
            "head_tilt",
            "x",
            "theta",
        ]
        self.info = {
            "total_tasks": 2,
            "features": {
                "action": {"shape": [16], "names": list(self.all_names)},
                "observation.state": {"shape": [16], "names": list(self.all_names)},
                "observation.images.head": {"dtype": "video"},
                "observation.images.left_wrist": {"dtype": "video"},
                "observation.images.right_wrist": {"dtype": "video"},
            },
        }

    def test_info_keeps_only_right_arm_and_retained_cameras(self) -> None:
        derived = _derive_info(self.info)

        self.assertEqual(derived["total_tasks"], 1)
        self.assertEqual(derived["features"]["action"]["shape"], [6])
        self.assertEqual(derived["features"]["action"]["names"], RIGHT_ARM_NAMES)
        self.assertNotIn("observation.images.left_wrist", derived["features"])
        self.assertIn("observation.images.head", derived["features"])
        self.assertIn("observation.images.right_wrist", derived["features"])
        self.assertEqual(self.info["features"]["action"]["shape"], [16])

    def test_global_stats_are_sliced_and_task_is_zeroed(self) -> None:
        stats = {
            "action": {"min": list(range(16)), "count": [100]},
            "observation.state": {"mean": list(range(16)), "count": [100]},
            "observation.images.left_wrist": {"count": [10]},
            "observation.images.head": {"count": [10]},
            "task_index": {"min": [0], "max": [1], "mean": [0.7], "count": [100]},
        }
        indices = _feature_indices(self.info, "action", RIGHT_ARM_NAMES)

        derived = _derive_global_stats(
            stats,
            action_indices=indices,
            state_indices=indices,
        )

        self.assertEqual(derived["action"]["min"], list(range(6, 12)))
        self.assertEqual(derived["observation.state"]["mean"], list(range(6, 12)))
        self.assertEqual(derived["task_index"]["max"], [0])
        self.assertEqual(derived["task_index"]["count"], [100])
        self.assertNotIn("observation.images.left_wrist", derived)


if __name__ == "__main__":
    unittest.main()
