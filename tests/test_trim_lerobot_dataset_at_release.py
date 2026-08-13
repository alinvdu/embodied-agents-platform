import unittest

import numpy as np
import pyarrow as pa

from scripts.trim_lerobot_dataset_at_release import (
    _find_sustained_run_start,
    _sustained_run_starts,
    detect_release_cuts,
)


def _table_for_gripper(gripper: list[float]) -> pa.Table:
    actions = np.zeros((len(gripper), 6), dtype=np.float32)
    actions[:, 5] = gripper
    action_array = pa.FixedSizeListArray.from_arrays(
        pa.array(actions.reshape(-1), type=pa.float32()),
        list_size=6,
    )
    return pa.table(
        {
            "action": action_array,
            "episode_index": pa.array(np.zeros(len(gripper), dtype=np.int64)),
            "frame_index": pa.array(np.arange(len(gripper), dtype=np.int64)),
            "index": pa.array(np.arange(len(gripper), dtype=np.int64)),
        }
    )


class TrimAtReleaseTests(unittest.TestCase):
    def test_sustained_runs_ignore_short_glitches(self) -> None:
        mask = np.array([False, True, False, True, True, True, False])

        self.assertEqual(_sustained_run_starts(mask, start=0, hold_frames=3), [3])
        self.assertEqual(_find_sustained_run_start(mask, start=0, hold_frames=3), 3)

    def test_cut_keeps_one_second_after_fully_open(self) -> None:
        gripper = (
            [1.0] * 10
            + [44.0] * 10
            + [1.0] * 15
            + [31.0, 35.0, 41.0]
            + [44.0] * 50
            + [1.0] * 10
        )
        table = _table_for_gripper(gripper)

        cuts = detect_release_cuts(
            table,
            episode_indices=[0],
            gripper_index=5,
            fps=30,
            open_threshold=30,
            closed_threshold=10,
            fully_open_threshold=40,
            hold_frames=3,
            post_release_s=1.0,
        )

        cut = cuts[0]
        self.assertEqual(cut.first_open_frame, 10)
        self.assertEqual(cut.grasp_close_frame, 20)
        self.assertEqual(cut.release_start_frame, 35)
        self.assertEqual(cut.fully_open_frame, 37)
        self.assertEqual(cut.reclose_start_frame, 88)
        self.assertEqual(cut.cut_frame_exclusive, 68)
        self.assertEqual(cut.retained_frames_after_fully_open, 30)
        self.assertEqual(gripper[cut.cut_frame_exclusive - 1], 44.0)

    def test_cut_stops_before_reset_reclose(self) -> None:
        gripper = [1.0] * 5 + [44.0] * 5 + [1.0] * 5 + [44.0] * 20 + [20.0] * 5 + [1.0] * 5

        cuts = detect_release_cuts(
            _table_for_gripper(gripper),
            episode_indices=[0],
            gripper_index=5,
            fps=30,
            open_threshold=30,
            closed_threshold=10,
            fully_open_threshold=40,
            hold_frames=3,
            post_release_s=1.0,
        )

        cut = cuts[0]
        self.assertEqual(cut.fully_open_frame, 15)
        self.assertEqual(cut.reclose_start_frame, 35)
        self.assertEqual(cut.cut_frame_exclusive, 35)
        self.assertEqual(gripper[cut.cut_frame_exclusive - 1], 44.0)

    def test_ambiguous_second_post_grasp_opening_is_rejected(self) -> None:
        gripper = [1.0] * 5 + [44.0] * 5 + [1.0] * 5 + [44.0] * 5 + [1.0] * 5 + [44.0] * 10

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            detect_release_cuts(
                _table_for_gripper(gripper),
                episode_indices=[0],
                gripper_index=5,
                fps=30,
                open_threshold=30,
                closed_threshold=10,
                fully_open_threshold=40,
                hold_frames=3,
                post_release_s=0.0,
            )


if __name__ == "__main__":
    unittest.main()
