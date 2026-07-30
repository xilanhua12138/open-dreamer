from __future__ import annotations

import unittest

import numpy as np

from dreamer.image_metrics import (
    edge_mask,
    psnr_from_squared_error,
    region_squared_error,
    temporal_change_mask,
)


class ImageRegionMetricTests(unittest.TestCase):
    def test_edge_mask_selects_crisp_boundary_not_flat_interior(self) -> None:
        target = np.zeros((1, 1, 4, 4, 3), dtype=np.uint8)
        target[:, :, :, 2:] = 255

        mask = edge_mask(target, threshold=16 / 255)

        self.assertTrue(mask[0, 0, 1, 1])
        self.assertFalse(mask[0, 0, 1, 3])

    def test_temporal_change_mask_excludes_first_frame_and_detects_motion(self) -> None:
        target = np.zeros((1, 3, 4, 4, 3), dtype=np.uint8)
        target[:, 1, 1, 1] = 255
        target[:, 2, 1, 2] = 255

        mask = temporal_change_mask(target, threshold=16 / 255)

        self.assertFalse(mask[:, 0].any())
        self.assertTrue(mask[0, 1, 1, 1])
        self.assertTrue(mask[0, 2, 1, 1])
        self.assertTrue(mask[0, 2, 1, 2])

    def test_region_psnr_uses_only_selected_pixels(self) -> None:
        target = np.zeros((1, 1, 2, 2, 3), dtype=np.uint8)
        prediction = target.copy()
        prediction[0, 0, 0, 0] = 255
        mask = np.zeros((1, 1, 2, 2), dtype=bool)
        mask[0, 0, 0, 0] = True

        squared_error, count = region_squared_error(prediction, target, mask)
        psnr = psnr_from_squared_error(squared_error, count)

        self.assertEqual(count, 3)
        self.assertAlmostEqual(squared_error, 3.0)
        self.assertAlmostEqual(psnr, 0.0)

    def test_region_squared_error_rejects_empty_mask(self) -> None:
        target = np.zeros((1, 1, 2, 2, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "no pixels"):
            region_squared_error(target, target, np.zeros(target.shape[:-1], bool))


if __name__ == "__main__":
    unittest.main()
