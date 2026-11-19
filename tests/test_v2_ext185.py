from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch

from pvfs.v2_sealed_and_ext185 import (
    aggregate_generic_episodes,
    build_ext185_episodes,
    load_ext185_records,
    run_v2_episode_ensemble,
)
from pvfs.v2_decoder import SCFBDecoder
from pvfs.v2_pipeline import V2_VARIANTS
from pvfs.freq_features import extract_freq_pyramid
from pvfs.fast_bank import build_pairwise_topk_cache
from pvfs.iter1_methods import _to_tokens
from pvfs.iter3_methods import compute_dino_affinity_weights
from pvfs.data import area_target_32
from pvfs.metrics import evaluate_episode


class V2SealedAndExt185Tests(unittest.TestCase):
    def test_load_ext185_episodes_and_ensemble_decoder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "PV-512-MERGED-185"
            (root / "images").mkdir(parents=True)
            (root / "labels").mkdir(parents=True)

            # Create 4 synthetic BMP samples (3 pos + 1 pure neg)
            for idx in range(1, 5):
                rgb = np.full((512, 512, 3), 120, dtype=np.uint8)
                lbl = np.zeros((512, 512), dtype=np.uint8)
                if idx <= 3:
                    lbl[120:240, 120:240] = 255
                    rgb[120:240, 120:240, 2] = 210
                Image.fromarray(rgb).save(root / "images" / f"PV-00000{idx}.bmp")
                Image.fromarray(lbl).save(root / "labels" / f"PV-00000{idx}.bmp")

            recs, rgbs, masks = load_ext185_records(root / "images", root / "labels")
            self.assertEqual(len(recs), 4)
            self.assertEqual(sum(r["positive"] for r in recs), 3)
            self.assertEqual(sum(not r["positive"] for r in recs), 1)

            dev_mock = [
                {"id": "dev/0", "site": "s0", "positive": True},
                {"id": "dev/1", "site": "s1", "positive": True},
            ]
            eps = build_ext185_episodes(dev_mock, recs, ks=(1,), num_episodes=2)
            self.assertEqual(len(eps["cross_dataset"]["1"]), 2)
            self.assertEqual(len(eps["in_domain"]["1"]), 2)
            self.assertNotIn(eps["in_domain"]["1"][0]["support_ids"][0], eps["in_domain"]["1"][0]["query_ids"])

            # Verify ensemble V2 evaluation + generic aggregator with 0 or 1 negatives
            torch.manual_seed(99)
            c = 64
            pv_vec, bg_vec = torch.randn(c), torch.randn(c)
            toks, affs, rgb_ts, f512s, f64s = {}, {}, {}, {}, {}
            for r in recs:
                rid = r["id"]
                m64 = masks[rid].reshape(64, 8, 64, 8).mean(axis=(1, 3)) >= 0.5
                f64_t = bg_vec[:, None, None] + torch.randn(c, 64, 64) * 0.08
                if m64.any():
                    f64_t[:, torch.from_numpy(m64)] = pv_vec[:, None] + torch.randn(c, int(m64.sum())) * 0.05
                t64 = _to_tokens(f64_t, "cpu")
                toks.setdefault("deep_64", {})[rid] = t64
                toks.setdefault("mid_64", {})[rid] = t64
                toks.setdefault("deep_32", {})[rid] = _to_tokens(torch.nn.functional.avg_pool2d(f64_t[None], 2, stride=2)[0], "cpu")
                affs[rid] = compute_dino_affinity_weights(t64, tau=0.12)
                rgb_ts[rid] = (torch.from_numpy(rgbs[rid]).float() / 255.0).half()
                p512, p64 = extract_freq_pyramid(rgbs[rid], device="cpu")
                f512s[rid] = p512
                f64s[rid] = p64

            all_ids = [r["id"] for r in recs]
            pos_ids = [r["id"] for r in recs if r["positive"]]
            caches = {
                k: build_pairwise_topk_cache(toks[k], masks, all_ids, pos_ids, grid_hw=(64 if "64" in k else 32))
                for k in ("deep_32", "mid_64", "deep_64")
            }
            dec_ens = [SCFBDecoder(in_ch=17).eval(), SCFBDecoder(in_ch=17).eval()]
            q_ids = [r["id"] for r in recs[1:]]
            q_items = [{**r, "mask": masks[r["id"]], "y32": area_target_32(masks[r["id"]])} for r in recs[1:]]
            full_var = next(v for v in V2_VARIANTS if v.name == "V2_Full_WithDecoder")
            p512_map, s32_map = run_v2_episode_ensemble(
                full_var, caches, toks, masks, affs, rgb_ts, f512s, f64s,
                [recs[0]["id"]], q_ids, decoders=dec_ens, device=torch.device("cpu"),
            )
            agg = aggregate_generic_episodes([evaluate_episode(q_items, p512_map, s32_map)])
            self.assertEqual(agg["macro"]["n_pos_sites"], 2)
            self.assertEqual(agg["macro"]["n_neg_sites"], 1)
            self.assertTrue(np.isfinite(agg["macro"]["dice"]))


if __name__ == "__main__":
    unittest.main()
