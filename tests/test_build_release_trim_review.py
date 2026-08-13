import unittest

from scripts.build_release_trim_review import _parse_episode_selection


class ReleaseTrimReviewTests(unittest.TestCase):
    def test_episode_selection_accepts_ranges(self) -> None:
        self.assertEqual(
            _parse_episode_selection("0,2,4-6", set(range(8))),
            [0, 2, 4, 5, 6],
        )

    def test_episode_selection_rejects_unknown_episode(self) -> None:
        with self.assertRaisesRegex(ValueError, "not present"):
            _parse_episode_selection("9", {0, 1, 2})

    def test_episode_selection_defaults_to_all_sorted(self) -> None:
        self.assertEqual(_parse_episode_selection(None, {4, 1, 3}), [1, 3, 4])


if __name__ == "__main__":
    unittest.main()
