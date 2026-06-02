import base64
import io
import json
import urllib.error
import unittest
from unittest.mock import patch

from xlerobot_agent import object_detection
from xlerobot_agent.object_detection import ObjectDetectorConfig, detect_object_in_image


class FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class ObjectDetectionTest(unittest.TestCase):
    def setUp(self) -> None:
        object_detection._REPLICATE_VERSION_CACHE.clear()

    def test_none_provider_reports_not_configured(self) -> None:
        result = detect_object_in_image(
            config=ObjectDetectorConfig(provider="none"),
            image_data_url="data:image/png;base64,cG5n",
            object_label="coke can",
            shot_id="shot_1",
        )

        self.assertEqual(result["status"], "not_configured")

    def test_mock_provider_matches(self) -> None:
        result = detect_object_in_image(
            config=ObjectDetectorConfig(provider="mock"),
            image_data_url="data:image/png;base64,cG5n",
            object_label="coke can",
            shot_id="shot_1",
        )

        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["selected_detection"]["label"], "coke can")

    def test_replicate_grounding_dino_normalizes_detections(self) -> None:
        calls = []

        def fake_urlopen(request, timeout=0):
            calls.append(request)
            if request.full_url.endswith("/v1/models/adirik/grounding-dino"):
                return FakeHTTPResponse({"latest_version": {"id": "version-123"}})
            if request.full_url.endswith("/v1/predictions"):
                self.assertIn("Robot42", request.headers.get("User-agent", ""))
                body = json.loads(request.data.decode("utf-8"))
                self.assertEqual(body["version"], "version-123")
                self.assertEqual(body["input"]["query"], "coke can")
                return FakeHTTPResponse(
                    {
                        "id": "prediction-123",
                        "status": "succeeded",
                        "output": {
                            "detections": [
                                {
                                    "label": "coke can",
                                    "score": 0.81,
                                    "bbox": [10, 20, 110, 220],
                                },
                                {
                                    "label": "low score",
                                    "score": 0.05,
                                    "bbox": [1, 2, 3, 4],
                                },
                            ],
                            "result_image": "https://replicate.local/result.jpg",
                        },
                    }
                )
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.object_detection.urllib.request.urlopen", side_effect=fake_urlopen):
            result = detect_object_in_image(
                config=ObjectDetectorConfig(
                    provider="replicate_grounding_dino",
                    api_key="token",
                    min_confidence=0.25,
                ),
                image_data_url="data:image/png;base64,cG5n",
                object_label="coke can",
                shot_id="shot_1",
                image_path="/tmp/shot.png",
            )

        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["selected_detection"]["bbox_xyxy"], [10.0, 20.0, 110.0, 220.0])
        self.assertEqual(result["selected_detection"]["confidence"], 0.81)
        self.assertEqual(result["provider_result_image_url"], "https://replicate.local/result.jpg")
        self.assertEqual(len(calls), 2)

    def test_replicate_center_box_coordinates_are_converted_to_xyxy(self) -> None:
        detections = object_detection._normalize_replicate_detections(
            [{"class": "can", "confidence": 0.7, "x": 100, "y": 80, "width": 40, "height": 20}],
            shot_id="shot_2",
            object_label="can",
            image_path=None,
            min_confidence=0.25,
        )

        self.assertEqual(detections[0]["bbox_xyxy"], [80.0, 70.0, 120.0, 90.0])

    def test_default_replicate_min_confidence_discards_weak_matches(self) -> None:
        def fake_urlopen(request, timeout=0):
            if request.full_url.endswith("/v1/models/adirik/grounding-dino"):
                return FakeHTTPResponse({"latest_version": {"id": "version-123"}})
            if request.full_url.endswith("/v1/predictions"):
                return FakeHTTPResponse(
                    {
                        "id": "prediction-123",
                        "status": "succeeded",
                        "output": {
                            "detections": [
                                {"label": "small bottle", "score": 0.53, "bbox": [10, 20, 110, 220]},
                            ],
                        },
                    }
                )
            raise AssertionError(f"unexpected URL {request.full_url}")

        with patch("xlerobot_agent.object_detection.urllib.request.urlopen", side_effect=fake_urlopen):
            result = detect_object_in_image(
                config=ObjectDetectorConfig(
                    provider="replicate_grounding_dino",
                    api_key="token",
                ),
                image_data_url="data:image/png;base64,cG5n",
                object_label="small bottle",
                shot_id="shot_weak",
            )

        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["detections"], [])
        self.assertIn("0.55", result["reason"])

    def test_replicate_preprocess_resizes_image_and_maps_boxes_back_to_source_pixels(self) -> None:
        if object_detection.Image is None:
            self.skipTest("Pillow is unavailable")
        image = object_detection.Image.new("RGB", (200, 100), (255, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

        resized_url, metadata = object_detection._prepare_replicate_image_data_url(
            data_url,
            max_edge_px=100,
            jpeg_quality=80,
        )
        detections = object_detection._normalize_replicate_detections(
            [{"label": "red thing", "confidence": 0.9, "bbox": [10, 20, 50, 40]}],
            shot_id="shot_3",
            object_label="red thing",
            image_path=None,
            min_confidence=0.25,
            image_preprocess=metadata,
        )

        self.assertTrue(resized_url.startswith("data:image/jpeg;base64,"))
        self.assertEqual(metadata["source_width"], 200)
        self.assertEqual(metadata["source_height"], 100)
        self.assertEqual(metadata["sent_width"], 100)
        self.assertEqual(metadata["sent_height"], 50)
        self.assertEqual(detections[0]["bbox_xyxy"], [20.0, 40.0, 100.0, 80.0])

    def test_replicate_retries_rate_limited_prediction_creation(self) -> None:
        prediction_calls = 0

        def fake_urlopen(request, timeout=0):
            nonlocal prediction_calls
            if request.full_url.endswith("/v1/models/adirik/grounding-dino"):
                return FakeHTTPResponse({"latest_version": {"id": "version-123"}})
            if request.full_url.endswith("/v1/predictions"):
                prediction_calls += 1
                if prediction_calls == 1:
                    raise urllib.error.HTTPError(
                        request.full_url,
                        429,
                        "Too Many Requests",
                        {"Retry-After": "0"},
                        io.BytesIO(b'{"retry_after": 0}'),
                    )
                return FakeHTTPResponse(
                    {
                        "id": "prediction-123",
                        "status": "succeeded",
                        "output": {
                            "detections": [
                                {"label": "bottle", "score": 0.8, "bbox": [1, 2, 3, 4]},
                            ],
                        },
                    }
                )
            raise AssertionError(f"unexpected URL {request.full_url}")

        with (
            patch("xlerobot_agent.object_detection.urllib.request.urlopen", side_effect=fake_urlopen),
            patch("xlerobot_agent.object_detection.time.sleep") as sleep,
        ):
            result = detect_object_in_image(
                config=ObjectDetectorConfig(
                    provider="replicate_grounding_dino",
                    api_key="token",
                    min_confidence=0.25,
                ),
                image_data_url="data:image/png;base64,cG5n",
                object_label="bottle",
                shot_id="shot_retry",
            )

        self.assertEqual(result["status"], "matched")
        self.assertEqual(prediction_calls, 2)
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
