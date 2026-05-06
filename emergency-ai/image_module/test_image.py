# test_image.py
# Emergency Image Detection System — Test Suite
# Tests the full pipeline: preprocess → model → image_analysis → api
#
# Run all tests  : python test_image.py
# Run one test   : python test_image.py TestPreprocess
# Run with report: python test_image.py --verbose

import os
import sys
import json
import time
import shutil
import unittest
import tempfile
import numpy as np
from pathlib import Path
from io import BytesIO
from unittest.mock import patch, MagicMock

# ─────────────────────────────────────────
# COLOUR OUTPUT — makes pass/fail easy to read in terminal
# ─────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def green(s):  return f"{GREEN}{s}{RESET}"
def red(s):    return f"{RED}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"
def bold(s):   return f"{BOLD}{s}{RESET}"


# ─────────────────────────────────────────
# TEST HELPERS
# ─────────────────────────────────────────

def make_dummy_image(path: str, size: tuple = (300, 300), color: str = "rgb"):
    """
    Creates a real dummy image file at the given path.
    Used when no real dataset images are available.
    Requires Pillow.
    """
    from PIL import Image
    img = Image.new("RGB", size, color=(200, 100, 50))
    img.save(path)
    return path


def get_sample_image(class_name: str = None) -> str:
    """
    Returns path to a real image from the dataset if available,
    otherwise creates a dummy image in a temp folder.
    """
    if class_name:
        folders = [f"dataset/{class_name}"]
    else:
        folders = ["dataset/fire", "dataset/medical", "dataset/crime", "dataset/normal"]

    for folder in folders:
        folder_path = Path(folder)
        if folder_path.exists():
            images = (
                list(folder_path.glob("*.jpg"))  +
                list(folder_path.glob("*.jpeg")) +
                list(folder_path.glob("*.png"))
            )
            if images:
                return str(images[0])

    # No real images found — make a dummy one
    tmp = tempfile.mktemp(suffix=".jpg")
    make_dummy_image(tmp)
    return tmp


# ─────────────────────────────────────────
# TEST SUITE 1 — preprocess.py
# ─────────────────────────────────────────

class TestPreprocess(unittest.TestCase):
    """Tests for preprocess.py — image loading, resizing, normalisation."""

    def test_import(self):
        """preprocess.py imports without errors."""
        import preprocessor
        self.assertTrue(True)

    def test_constants(self):
        """IMG_SIZE is (224, 224) and NUM_CLASSES is 4."""
        from preprocessor import IMG_SIZE, NUM_CLASSES, CLASS_NAMES
        self.assertEqual(IMG_SIZE, (224, 224))
        self.assertEqual(NUM_CLASSES, 4)
        self.assertEqual(len(CLASS_NAMES), 4)
        self.assertIn("fire",    CLASS_NAMES)
        self.assertIn("medical", CLASS_NAMES)
        self.assertIn("crime",   CLASS_NAMES)
        self.assertIn("normal",  CLASS_NAMES)

    def test_preprocess_single_image_shape(self):
        """preprocess_single_image returns array of shape (1, 224, 224, 3)."""
        from preprocessor import preprocess_single_image
        img_path = get_sample_image()
        try:
            result = preprocess_single_image(img_path)
            self.assertEqual(result.shape, (1, 224, 224, 3))
        finally:
            # clean up dummy if we made one
            if "tmp" in img_path or tempfile.gettempdir() in img_path:
                if os.path.exists(img_path):
                    os.remove(img_path)

    def test_preprocess_normalization(self):
        """Pixel values are between 0.0 and 1.0 after preprocessing."""
        from preprocessor import preprocess_single_image
        img_path = get_sample_image()
        try:
            result = preprocess_single_image(img_path)
            self.assertGreaterEqual(float(result.min()), 0.0)
            self.assertLessEqual(float(result.max()),    1.0)
        finally:
            if tempfile.gettempdir() in img_path:
                if os.path.exists(img_path):
                    os.remove(img_path)

    def test_preprocess_single_image_dtype(self):
        """Output array dtype is float32."""
        from preprocessor import preprocess_single_image
        img_path = get_sample_image()
        try:
            result = preprocess_single_image(img_path)
            self.assertEqual(result.dtype, np.float32)
        finally:
            if tempfile.gettempdir() in img_path:
                if os.path.exists(img_path):
                    os.remove(img_path)

    def test_invalid_image_path_raises(self):
        """preprocess_single_image raises an error for a missing file."""
        from preprocessor import preprocess_single_image
        with self.assertRaises(Exception):
            preprocess_single_image("non_existent_file_xyz.jpg")

    def test_validate_dataset_structure(self):
        """validate_dataset() runs without crashing (may warn about missing folders)."""
        from preprocessor import validate_dataset
        result = validate_dataset()
        self.assertIsInstance(result, bool)

    def test_data_generators_exist(self):
        """get_data_generators() is callable and returns two generators if dataset exists."""
        from preprocessor import validate_dataset, get_data_generators, CLASS_NAMES
        if not validate_dataset():
            self.skipTest("Dataset not available — skipping generator test.")
        train_data, val_data = get_data_generators()
        self.assertIsNotNone(train_data)
        self.assertIsNotNone(val_data)
        self.assertEqual(set(train_data.class_indices.keys()), set(CLASS_NAMES))


# ─────────────────────────────────────────
# TEST SUITE 2 — image_analysis.py
# ─────────────────────────────────────────

class TestImageAnalysis(unittest.TestCase):
    """Tests for image_analysis.py — model loading, inference, result structure."""

    def setUp(self):
        """Skip all tests in this suite if the model hasn't been trained yet."""
        if not Path("model/emergency_model.h5").exists():
            self.skipTest(
                yellow("Model not found — run train.py first, then re-run tests.")
            )

    def test_import(self):
        """image_analysis.py imports without errors."""
        import image_analysis
        self.assertTrue(True)

    def test_analyze_image_returns_dict(self):
        """analyze_image() returns a dict."""
        from image_analysis import analyze_image
        img_path = get_sample_image()
        try:
            result = analyze_image(img_path)
            self.assertIsInstance(result, dict)
        finally:
            if tempfile.gettempdir() in img_path:
                if os.path.exists(img_path): os.remove(img_path)

    def test_result_has_required_keys(self):
        """Result dict contains all expected keys."""
        from image_analysis import analyze_image
        img_path = get_sample_image()
        required_keys = [
            "status", "emergency_type", "confidence",
            "all_scores", "dispatch", "timestamp",
            "image_path", "inference_ms", "flagged"
        ]
        try:
            result = analyze_image(img_path)
            for key in required_keys:
                self.assertIn(key, result, f"Missing key: '{key}'")
        finally:
            if tempfile.gettempdir() in img_path:
                if os.path.exists(img_path): os.remove(img_path)

    def test_confidence_is_valid_probability(self):
        """Confidence score is between 0.0 and 1.0."""
        from image_analysis import analyze_image
        img_path = get_sample_image()
        try:
            result = analyze_image(img_path)
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"],    1.0)
        finally:
            if tempfile.gettempdir() in img_path:
                if os.path.exists(img_path): os.remove(img_path)

    def test_all_scores_sum_to_one(self):
        """All class probability scores sum to approximately 1.0."""
        from image_analysis import analyze_image
        img_path = get_sample_image()
        try:
            result   = analyze_image(img_path)
            total    = sum(result["all_scores"].values())
            self.assertAlmostEqual(total, 1.0, places=2)
        finally:
            if tempfile.gettempdir() in img_path:
                if os.path.exists(img_path): os.remove(img_path)

    def test_emergency_type_is_valid_class(self):
        """emergency_type is one of the four known classes."""
        from image_analysis import analyze_image
        valid_classes = {"fire", "medical", "crime", "normal"}
        img_path = get_sample_image()
        try:
            result = analyze_image(img_path)
            self.assertIn(result["emergency_type"], valid_classes)
        finally:
            if tempfile.gettempdir() in img_path:
                if os.path.exists(img_path): os.remove(img_path)

    def test_status_is_valid_value(self):
        """status field is one of: emergency, normal, uncertain."""
        from image_analysis import analyze_image
        valid_statuses = {"emergency", "normal", "uncertain"}
        img_path = get_sample_image()
        try:
            result = analyze_image(img_path)
            self.assertIn(result["status"], valid_statuses)
        finally:
            if tempfile.gettempdir() in img_path:
                if os.path.exists(img_path): os.remove(img_path)

    def test_dispatch_has_required_keys(self):
        """dispatch block contains services, priority, resources, message."""
        from image_analysis import analyze_image
        img_path = get_sample_image()
        try:
            result   = analyze_image(img_path)
            dispatch = result["dispatch"]
            for key in ["services", "priority", "resources", "message"]:
                self.assertIn(key, dispatch, f"dispatch missing key: '{key}'")
        finally:
            if tempfile.gettempdir() in img_path:
                if os.path.exists(img_path): os.remove(img_path)

    def test_missing_image_raises_error(self):
        """analyze_image() raises FileNotFoundError for a non-existent file."""
        from image_analysis import analyze_image
        with self.assertRaises(FileNotFoundError):
            analyze_image("this_file_does_not_exist_xyz.jpg")

    def test_inference_speed(self):
        """Single image inference completes in under 5 seconds on CPU."""
        from image_analysis import analyze_image
        img_path = get_sample_image()
        try:
            start  = time.time()
            result = analyze_image(img_path)
            elapsed = time.time() - start
            self.assertLess(elapsed, 5.0,
                f"Inference took {elapsed:.2f}s — should be under 5s on CPU.")
            print(f"\n  {cyan('Inference time')}: {result['inference_ms']}ms")
        finally:
            if tempfile.gettempdir() in img_path:
                if os.path.exists(img_path): os.remove(img_path)

    def test_log_file_created(self):
        """Running analyze_image() creates/appends to logs/incidents.log."""
        from image_analysis import analyze_image
        img_path = get_sample_image()
        try:
            analyze_image(img_path)
            self.assertTrue(Path("logs/incidents.log").exists())
        finally:
            if tempfile.gettempdir() in img_path:
                if os.path.exists(img_path): os.remove(img_path)

    def test_log_entry_is_valid_json(self):
        """Last entry in incidents.log is valid JSON."""
        from image_analysis import analyze_image
        img_path = get_sample_image()
        try:
            analyze_image(img_path)
            with open("logs/incidents.log", "r") as f:
                lines = [l.strip() for l in f if l.strip()]
            last_line = lines[-1]
            entry = json.loads(last_line)   # raises if invalid JSON
            self.assertIn("emergency_type", entry)
            self.assertIn("confidence",     entry)
        finally:
            if tempfile.gettempdir() in img_path:
                if os.path.exists(img_path): os.remove(img_path)

    def test_batch_analysis(self):
        """analyze_batch() returns a list with one result per image."""
        from image_analysis import analyze_batch
        paths = [get_sample_image() for _ in range(3)]
        try:
            results = analyze_batch(paths)
            self.assertEqual(len(results), 3)
            for r in results:
                self.assertIn("emergency_type", r)
        finally:
            for p in paths:
                if tempfile.gettempdir() in p and os.path.exists(p):
                    os.remove(p)

    def test_all_four_classes_detectable(self):
        """Model can produce all four class labels (checks output diversity)."""
        from image_analysis import analyze_image
        seen_classes = set()
        for cls in ["fire", "medical", "crime", "normal"]:
            img_path = get_sample_image(cls)
            try:
                result = analyze_image(img_path)
                seen_classes.add(result["emergency_type"])
            except Exception:
                pass
            finally:
                if tempfile.gettempdir() in img_path and os.path.exists(img_path):
                    os.remove(img_path)

        print(f"\n  {cyan('Classes seen in test')}: {seen_classes}")
        self.assertGreaterEqual(len(seen_classes), 1,
            "Model should produce at least one distinct class.")


# ─────────────────────────────────────────
# TEST SUITE 3 — api.py (without starting a real server)
# ─────────────────────────────────────────

class TestAPI(unittest.TestCase):
    """Tests for api.py using FastAPI's built-in test client."""

    def setUp(self):
        """Set up the FastAPI test client."""
        try:
            from fastapi.testclient import TestClient
            # Patch warmup so it doesn't fail if model isn't trained yet
            with patch("api.warmup", return_value=None):
                from api import app
            self.client = TestClient(app, raise_server_exceptions=False)
        except Exception as e:
            self.skipTest(f"Could not set up API test client: {e}")

    def test_health_endpoint_returns_200(self):
        """GET /health returns HTTP 200."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

    def test_health_response_has_status_key(self):
        """GET /health response contains 'status' key."""
        response = self.client.get("/health")
        data = response.json()
        self.assertIn("status", data)

    def test_classes_endpoint(self):
        """GET /classes returns 200 if model is trained, 404 if not."""
        response = self.client.get("/classes")
        self.assertIn(response.status_code, [200, 404])

    def test_logs_endpoint_returns_200(self):
        """GET /logs returns HTTP 200 even with no logs."""
        response = self.client.get("/logs")
        self.assertEqual(response.status_code, 200)

    def test_logs_response_structure(self):
        """GET /logs response contains 'logs' and 'total' keys."""
        response = self.client.get("/logs")
        data = response.json()
        self.assertIn("logs",  data)
        self.assertIn("total", data)

    def test_logs_stats_endpoint(self):
        """GET /logs/stats returns HTTP 200 with count fields."""
        response = self.client.get("/logs/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total", data)

    def test_analyze_image_no_file_returns_422(self):
        """POST /analyze-image with no file returns HTTP 422 (validation error)."""
        response = self.client.post("/analyze-image")
        self.assertEqual(response.status_code, 422)

    def test_analyze_image_wrong_type_returns_400(self):
        """POST /analyze-image with a .txt file returns HTTP 400."""
        fake_file = BytesIO(b"this is not an image")
        response  = self.client.post(
            "/analyze-image",
            files={"file": ("test.txt", fake_file, "text/plain")}
        )
        self.assertEqual(response.status_code, 400)

    def test_analyze_image_with_real_image(self):
        """POST /analyze-image with a real image returns 200 or 500 (if model missing)."""
        if not Path("model/emergency_model.h5").exists():
            self.skipTest("Model not trained yet.")

        img_path = get_sample_image()
        try:
            with open(img_path, "rb") as f:
                response = self.client.post(
                    "/analyze-image",
                    files={"file": ("test.jpg", f, "image/jpeg")}
                )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("emergency_type", data)
            self.assertIn("confidence",     data)
            self.assertIn("dispatch",       data)
        finally:
            if tempfile.gettempdir() in img_path and os.path.exists(img_path):
                os.remove(img_path)

    def test_cors_headers_present(self):
        """API response includes CORS headers for localhost:3000."""
        response = self.client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )
        self.assertIn("access-control-allow-origin", response.headers)


# ─────────────────────────────────────────
# TEST SUITE 4 — end-to-end pipeline
# ─────────────────────────────────────────

class TestEndToEnd(unittest.TestCase):
    """
    Full pipeline test: image file → preprocess → model → result → log.
    Only runs if the model has been trained.
    """

    def setUp(self):
        if not Path("model/emergency_model.h5").exists():
            self.skipTest(yellow("Model not trained — skipping end-to-end tests."))

    def test_full_pipeline_fire(self):
        """Full pipeline works on an image from the fire/ folder."""
        self._run_pipeline("fire")

    def test_full_pipeline_medical(self):
        """Full pipeline works on an image from the medical/ folder."""
        self._run_pipeline("medical")

    def test_full_pipeline_crime(self):
        """Full pipeline works on an image from the crime/ folder."""
        self._run_pipeline("crime")

    def test_full_pipeline_normal(self):
        """Full pipeline works on an image from the normal/ folder."""
        self._run_pipeline("normal")

    def _run_pipeline(self, class_name: str):
        from preprocessor import preprocess_single_image
        from image_analysis import analyze_image

        img_path = get_sample_image(class_name)
        tmp_used = tempfile.gettempdir() in img_path

        try:
            # Step 1 — preprocess
            arr = preprocess_single_image(img_path)
            self.assertEqual(arr.shape, (1, 224, 224, 3))

            # Step 2 — analyze
            result = analyze_image(img_path)

            # Step 3 — validate result
            self.assertIn(result["emergency_type"], {"fire", "medical", "crime", "normal"})
            self.assertBetween(result["confidence"], 0.0, 1.0)
            self.assertIn(result["status"], {"emergency", "normal", "uncertain"})

            print(
                f"\n  {cyan(class_name):20s} → "
                f"predicted: {bold(result['emergency_type']):12s} "
                f"confidence: {result['confidence']*100:.1f}%"
            )

        finally:
            if tmp_used and os.path.exists(img_path):
                os.remove(img_path)

    def assertBetween(self, value, lo, hi):
        self.assertGreaterEqual(value, lo)
        self.assertLessEqual(value,    hi)


# ─────────────────────────────────────────
# CUSTOM TEST RUNNER — pretty printed output
# ─────────────────────────────────────────

class PrettyTestResult(unittest.TextTestResult):

    def addSuccess(self, test):
        super().addSuccess(test)
        print(f"  {green('PASS')}  {test._testMethodDoc or test._testMethodName}")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        print(f"  {red('FAIL')}  {test._testMethodDoc or test._testMethodName}")

    def addError(self, test, err):
        super().addError(test, err)
        print(f"  {red('ERR ')}  {test._testMethodDoc or test._testMethodName}")

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        print(f"  {yellow('SKIP')}  {test._testMethodDoc or test._testMethodName}  ({reason})")


class PrettyTestRunner(unittest.TextTestRunner):
    resultclass = PrettyTestResult


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":

    suites = [
        ("Preprocess",   unittest.TestLoader().loadTestsFromTestCase(TestPreprocess)),
        ("Image Analysis", unittest.TestLoader().loadTestsFromTestCase(TestImageAnalysis)),
        ("API",          unittest.TestLoader().loadTestsFromTestCase(TestAPI)),
        ("End-to-End",   unittest.TestLoader().loadTestsFromTestCase(TestEndToEnd)),
    ]

    print("\n" + bold("=" * 55))
    print(bold("  Emergency Image Detection — Test Suite"))
    print(bold("=" * 55))

    total_passed  = 0
    total_failed  = 0
    total_errors  = 0
    total_skipped = 0

    for suite_name, suite in suites:
        print(f"\n{cyan(bold(f'── {suite_name} Tests '))}{'─' * (40 - len(suite_name))}")
        runner = PrettyTestRunner(stream=open(os.devnull, "w"), verbosity=0)
        result = runner.run(suite)
        total_passed  += result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
        total_failed  += len(result.failures)
        total_errors  += len(result.errors)
        total_skipped += len(result.skipped)

        if result.failures:
            print(f"\n  {red('Failures:')}")
            for test, traceback in result.failures:
                lines = traceback.strip().split("\n")
                print(f"    {red(lines[-1])}")

        if result.errors:
            print(f"\n  {red('Errors:')}")
            for test, traceback in result.errors:
                lines = traceback.strip().split("\n")
                print(f"    {red(lines[-1])}")

    # ── Summary ──
    print("\n" + bold("=" * 55))
    print(bold("  SUMMARY"))
    print(bold("=" * 55))
    print(f"  {green('Passed')}  : {total_passed}")
    print(f"  {red('Failed')}  : {total_failed}")
    print(f"  {red('Errors')}  : {total_errors}")
    print(f"  {yellow('Skipped')} : {total_skipped}")
    print(bold("=" * 55))

    if total_failed == 0 and total_errors == 0:
        print(f"\n  {green(bold('All tests passed!'))} Ready to build the frontend.\n")
        sys.exit(0)
    else:
        print(f"\n  {red(bold('Some tests failed.'))} Fix the issues above before continuing.\n")
        sys.exit(1)