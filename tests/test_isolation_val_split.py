"""Unit tests for sticky isolation val split (stable IoU across retrains)."""
from __future__ import annotations

import unittest

from app.isolation_finetune import _pair_stable_key, _sticky_train_val_split


class StickyValSplitTests(unittest.TestCase):
    def test_same_pairs_same_holdout(self) -> None:
        items = [{"i": i} for i in range(8)]
        keys = [f"idx:{i}" for i in range(8)]
        t1, v1, n1 = _sticky_train_val_split(items, keys=keys, val_split=0.25)
        t2, v2, n2 = _sticky_train_val_split(items, keys=keys, val_split=0.25)
        self.assertEqual(n1, n2)
        self.assertEqual([x["i"] for x in v1], [x["i"] for x in v2])
        self.assertEqual([x["i"] for x in t1], [x["i"] for x in t2])
        self.assertEqual(len(v1), max(1, int(8 * 0.25)))
        self.assertEqual(len(t1) + len(v1), 8)

    def test_adding_pair_keeps_prior_val_when_n_val_unchanged(self) -> None:
        keys = [f"idx:{i}" for i in range(4)]
        items = [{"i": i} for i in range(4)]
        _, v1, n1 = _sticky_train_val_split(items, keys=keys, val_split=0.2)
        self.assertEqual(n1, 1)
        val_ids = {x["i"] for x in v1}

        keys2 = keys + ["idx:4"]
        items2 = items + [{"i": 4}]
        _, v2, n2 = _sticky_train_val_split(items2, keys=keys2, val_split=0.2)
        self.assertEqual(n2, 1)
        # With n_val still 1, the single holdout stays the same lowest-hash pair.
        self.assertEqual({x["i"] for x in v2}, val_ids)

    def test_pair_stable_key_prefers_index(self) -> None:
        self.assertEqual(_pair_stable_key({"index": 3, "before": "/x/before.jpg"}), "idx:3")


if __name__ == "__main__":
    unittest.main()
