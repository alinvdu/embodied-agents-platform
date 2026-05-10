from __future__ import annotations

import math
import unittest

from xlerobot_playground.imu_yaw_filter import compensate_yaw_rate_for_pitch


class ImuYawFilterTests(unittest.TestCase):
    def test_pitch_compensation_corrects_gyro_y_projection(self) -> None:
        corrected = compensate_yaw_rate_for_pitch(
            math.cos(math.radians(30.0)),
            yaw_source="gyro_y",
            pitch_rad=math.radians(30.0),
            enabled=True,
            min_cos=0.25,
        )

        self.assertAlmostEqual(corrected, 1.0)

    def test_pitch_compensation_does_not_affect_other_sources(self) -> None:
        corrected = compensate_yaw_rate_for_pitch(
            0.5,
            yaw_source="gyro_z",
            pitch_rad=math.radians(30.0),
            enabled=True,
            min_cos=0.25,
        )

        self.assertAlmostEqual(corrected, 0.5)


if __name__ == "__main__":
    unittest.main()
