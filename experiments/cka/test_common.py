"""Run with: python -m unittest experiments.cka.test_common -v."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from experiments.cka.common import (
    ITSPMRepresentationCapture,
    cka_statistics,
    deterministic_indices,
    linear_cka,
    save_cka_result,
)
from models.ITSPM import ITSPM


def gram_cka_reference(x, y):
    """Independent double-centered Gram formula, not the feature implementation."""
    n = len(x)
    centering = np.eye(n) - np.ones((n, n)) / n
    k = centering @ (x @ x.T) @ centering
    l = centering @ (y @ y.T) @ centering
    return np.sum(k * l) / (np.linalg.norm(k, "fro") * np.linalg.norm(l, "fro"))


class CKAMathTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(101)
        self.x = self.rng.normal(size=(57, 8))
        self.y = self.rng.normal(size=(57, 13))

    def test_feature_matches_gram(self):
        self.assertAlmostEqual(linear_cka(self.x, self.y), gram_cka_reference(self.x, self.y), places=13)

    def test_invariances(self):
        score = linear_cka(self.x, self.y)
        qx = np.linalg.qr(self.rng.normal(size=(8, 8)))[0]
        qy = np.linalg.qr(self.rng.normal(size=(13, 13)))[0]
        changed = linear_cka(17 * (self.x @ qx) + 4, -3 * (self.y @ qy) + np.arange(13))
        self.assertAlmostEqual(score, changed, places=13)
        perm = self.rng.permutation(len(self.x))
        self.assertAlmostEqual(score, linear_cka(self.x[perm], self.y[perm]), places=13)
        self.assertAlmostEqual(linear_cka(self.x, self.x), 1.0, places=13)

    def test_joint_global_centering_and_pairing(self):
        y = self.x + self.rng.normal(scale=0.05, size=self.x.shape)
        global_score = linear_cka(np.concatenate(np.array_split(self.x, 7)), np.concatenate(np.array_split(y, 7)))
        self.assertAlmostEqual(global_score, linear_cka(self.x, y), places=13)
        self.assertLess(linear_cka(self.x, y[self.rng.permutation(len(y))]), global_score)

    def test_extreme_scalar_rescaling(self):
        self.assertAlmostEqual(linear_cka(self.x * 1e180, self.y * 1e-180), linear_cka(self.x, self.y), places=13)

    def test_degenerate_and_invalid_inputs(self):
        for x, y in [
            (np.zeros((5, 2)), np.ones((5, 3))),
            (np.tile([1.0, 2.0, 3.0], (7, 1)), self.x[:7]),
            (np.ones((1, 2)), np.ones((1, 3))),
            (np.array([[np.nan], [1.0]]), np.ones((2, 1))),
            (np.zeros((5, 2)), np.zeros((4, 2))),
        ]:
            with self.assertRaises(ValueError):
                linear_cka(x, y)
        undefined = cka_statistics(np.ones((5, 2)), self.x[:5])
        self.assertEqual(undefined["status"], "undefined")
        self.assertIsNone(undefined["linear_cka"])

    def test_sampling_and_artifact_alignment(self):
        ids = deterministic_indices(57, 11, seed=3)
        np.testing.assert_array_equal(ids, deterministic_indices(57, 11, seed=3))
        self.assertTrue(np.all(np.diff(ids) > 0))
        valid = np.ones(len(ids), dtype=bool)
        valid[2] = False
        with TemporaryDirectory() as tmp:
            report = save_cka_result(tmp, self.x[ids], self.y[ids], ids, {"seed": 3}, valid=valid)
            loaded = np.load(Path(tmp) / "representations.npz", allow_pickle=False)
            np.testing.assert_array_equal(loaded["sample_ids"], ids[valid])
            np.testing.assert_array_equal(loaded["first"], self.x[ids][valid])
            stored = json.loads((Path(tmp) / "results.json").read_text())
            self.assertEqual(stored, report)
            self.assertEqual(report["n_dropped_all_missing"], 1)
            self.assertAlmostEqual(report["linear_cka"], linear_cka(self.x[ids][valid], self.y[ids][valid]))


class CaptureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.model = ITSPM(SimpleNamespace(
            input_dim=2, d_model=8, dropout=0.1, n_ref_points=4,
            max_event_tokens=6, max_gap_tokens=4, n_mixer_layers=1, n_scales=2,
        )).eval()
        self.times = torch.linspace(0, 1, 9).expand(4, -1)
        self.values = torch.randn(4, 9, 2)
        self.mask = (torch.rand(4, 9, 2) > 0.3).float()
        self.mask[1] = 0
        self.values *= self.mask

    def test_capture_does_not_change_forward_and_matches_endpoints(self):
        with torch.no_grad():
            expected = self.model(self.times, self.values, self.mask)
            global_expected = self.model._encode(self.values, self.times, self.mask)[0]
            times3 = self.model._prepare_tp(self.times, self.mask)
            tmin, tmax = self.model.encoder.compute_time_range(times3, self.mask)
            events, _ = self.model.encoder.encode(self.values, times3, self.mask, tmin, tmax)
            expected_first = (events.double() * self.mask.double().unsqueeze(-1)).sum((1, 2)) / self.mask.double().sum((1, 2)).clamp(min=1).unsqueeze(-1)
            before_rng = torch.random.get_rng_state()
            with ITSPMRepresentationCapture(self.model) as capture:
                capture.begin_batch(self.mask)
                actual = self.model(self.times, self.values, self.mask)
                reps = capture.end_batch(last="forward")
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))
            torch.testing.assert_close(reps["first"], expected_first)
            torch.testing.assert_close(reps["last"], expected.double())
            torch.testing.assert_close(reps["shared_global"], global_expected.double())
            self.assertTrue(torch.equal(reps["shared_global"], global_expected.double()))
            self.assertEqual(reps["valid"].tolist(), [True, False, True, True])
            self.assertFalse(reps["first"].requires_grad)
            self.assertFalse(self.model.training)
            self.assertFalse(self.model.encoder.norm._forward_hooks)

    def test_forecasting_direct_encode_path_and_cleanup_on_exception(self):
        with torch.no_grad():
            expected = self.model.forecasting(self.times, self.values, self.times, self.mask)
            with ITSPMRepresentationCapture(self.model) as capture:
                capture.begin_batch(self.mask)
                actual = self.model.forecasting(self.times, self.values, self.times, self.mask)
                reps = capture.end_batch()
            self.assertTrue(torch.equal(actual, expected))
            self.assertNotIn("forward", reps)
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with ITSPMRepresentationCapture(self.model):
                raise RuntimeError("sentinel")
        self.assertFalse(self.model.interaction._forward_hooks)

    def test_missing_and_repeated_forward_are_rejected(self):
        with ITSPMRepresentationCapture(self.model) as capture:
            capture.begin_batch(self.mask)
            with self.assertRaisesRegex(RuntimeError, "Missing captured"):
                capture.end_batch()
            self.model(self.times, self.values, self.mask)
            with self.assertRaisesRegex(RuntimeError, "Multiple backbone"):
                self.model(self.times, self.values, self.mask)


if __name__ == "__main__":
    unittest.main()
