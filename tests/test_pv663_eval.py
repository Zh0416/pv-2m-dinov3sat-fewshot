from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from pvfs.fast_bank import build_pairwise_topk_cache
from pvfs.io import read_json
from pvfs.iter3_methods import _to_tokens_norm
from pvfs.pv663_eval import (
    build_pairwise_topk_cache_streaming,
    run_pv663_evaluation_from_features,
)


class PV663EvalTests(unittest.TestCase):
    def test_locked_split_and_episodes_spatial_isolation(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sp = read_json(root / "configs" / "pv663_split_lock.json")
        ep = read_json(root / "configs" / "pv663_episodes_lock.json")

        self.assertEqual(sp["summary"]["total_images"], 663)
        self.assertGreater(sp["min_support_query_distance_km"], 5.0)

        by_id = {r["id"]: r for r in sp["records"]}
        sup_sites = {r["site"] for r in sp["records"] if r["partition"] == "support_pool"}
        qry_sites = {r["site"] for r in sp["records"] if r["partition"] == "query_set"}
        self.assertEqual(len(sup_sites & qry_sites), 0)
        self.assertEqual(len(ep["episodes"]), 30)  # 10 episodes x K in {1, 3, 5}

        qry_set = set(ep["query_ids"])
        for e in ep["episodes"]:
            self.assertEqual(len(e["support_ids"]), e["k"])
            self.assertEqual(len(set(e["support_sites"])), e["k"])
            for sid in e["support_ids"]:
                self.assertEqual(by_id[sid]["partition"], "support_pool")
                self.assertNotIn(sid, qry_set)

    def test_streaming_cache_matches_fast_bank_and_end_to_end_pipeline(self) -> None:
        rng = np.random.default_rng(20261124)
        torch.manual_seed(20261124)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            feat_dir = tmp_p / "feats"
            out_dir = tmp_p / "out"
            feat_dir.mkdir(parents=True, exist_ok=True)

            # Create 6 synthetic images (3 support sites, 3 query sites) with clear PV square signal
            ids = [f"PV08_31000{i}_119000{i}.bmp" for i in range(6)]
            sup_ids_pool = ids[:3]
            qry_ids_pool = ids[3:]

            fg_vec_d = F.normalize(torch.randn(16), dim=0)
            bg_vec_d = F.normalize(torch.randn(16), dim=0)
            fg_vec_m = F.normalize(torch.randn(16), dim=0)
            bg_vec_m = F.normalize(torch.randn(16), dim=0)

            mask_by_id: dict[str, np.ndarray] = {}
            rgb_by_id: dict[str, np.ndarray] = {}
            tokens_mem: dict[str, dict[str, torch.Tensor]] = {"deep_32": {}, "deep_64": {}, "mid_64": {}}

            for idx, rid in enumerate(ids):
                mask = np.zeros((512, 512), dtype=np.uint8)
                y0, y1 = 128 + idx * 8, 256 + idx * 8
                x0, x1 = 128 + idx * 8, 256 + idx * 8
                mask[y0:y1, x0:x1] = 1
                mask_by_id[rid] = mask

                rgb = np.full((512, 512, 3), 180, dtype=np.uint8)
                rgb[y0:y1, x0:x1] = [40, 60, 95]
                rgb_by_id[rid] = rgb

                def _make_feat(hw: int, fg_v: torch.Tensor, bg_v: torch.Tensor) -> torch.Tensor:
                    p = 512 // hw
                    m_g = mask.reshape(hw, p, hw, p).mean(axis=(1, 3)) >= 0.5
                    fmap = torch.randn(16, hw, hw) * 0.08
                    fmap[:, m_g] += fg_v[:, None]
                    fmap[:, ~m_g] += bg_v[:, None]
                    return F.normalize(fmap, dim=0).half()

                d32 = _make_feat(32, fg_vec_d, bg_vec_d)
                d64 = _make_feat(64, fg_vec_d, bg_vec_d)
                m64 = _make_feat(64, fg_vec_m, bg_vec_m)
                torch.save({"deep_32": d32, "deep_64": d64, "mid_64": m64}, feat_dir / f"{Path(rid).stem}.pt")
                tokens_mem["deep_32"][rid] = _to_tokens_norm(d32, "cpu")
                tokens_mem["deep_64"][rid] = _to_tokens_norm(d64, "cpu")
                tokens_mem["mid_64"][rid] = _to_tokens_norm(m64, "cpu")

            # 1. Exact equivalence test: streaming cache == fast_bank cache
            c_ref = build_pairwise_topk_cache(
                tokens_mem["deep_64"], mask_by_id, ids, sup_ids_pool, grid_hw=64, core_purity=0.85, top_m=3
            )
            c_str = build_pairwise_topk_cache_streaming(
                feat_dir, "deep_64", mask_by_id, ids, sup_ids_pool, grid_hw=64, core_purity=0.85, top_m=3, device="cpu", chunk_imgs=2
            )
            for tkey in ("fg_core_top", "fg_all_top", "bg_top"):
                self.assertTrue(torch.equal(c_ref[tkey], c_str[tkey]), f"Mismatch in {tkey}")

            # 2. End-to-end synthetic run (K=1 and K=3, 2 episodes each)
            records = []
            for idx, rid in enumerate(ids):
                records.append(
                    {
                        "id": rid,
                        "site": f"site_{idx}",
                        "partition": "support_pool" if idx < 3 else "query_set",
                        "fold": None,
                        "positive": True,
                        "target_fraction": float(mask_by_id[rid].mean()),
                    }
                )
            split_p = {
                "min_support_query_distance_km": 10.0,
                "summary": {
                    "total_images": 6,
                    "total_sites": 6,
                    "support_pool_sites": 3,
                    "query_set_sites": 3,
                    "support_pool": {"total": 3},
                    "query_set": {"total": 3},
                },
                "records": records,
            }
            ep_p = {
                "episodes_per_k": 2,
                "unique_support_ids": sup_ids_pool,
                "query_ids": qry_ids_pool,
                "episodes": [
                    {"k": 1, "episode": 0, "support_sites": ["site_0"], "support_ids": [sup_ids_pool[0]]},
                    {"k": 1, "episode": 1, "support_sites": ["site_1"], "support_ids": [sup_ids_pool[1]]},
                    {"k": 3, "episode": 0, "support_sites": ["site_0", "site_1", "site_2"], "support_ids": sup_ids_pool},
                    {"k": 3, "episode": 1, "support_sites": ["site_0", "site_1", "site_2"], "support_ids": sup_ids_pool},
                ],
            }
            deform_cfg = Path(__file__).resolve().parents[1] / "configs" / "three_module_exp_m3_deformgrad.json"
            summary = run_pv663_evaluation_from_features(
                split_payload=split_p,
                episodes_payload=ep_p,
                feat_dir=feat_dir,
                rgb_512_by_id=rgb_by_id,
                mask_512_by_id=mask_by_id,
                deform_config_json=deform_cfg,
                output_dir=out_dir,
                device="cpu",
                num_workers=1,
                n_boot=200,
            )
            self.assertTrue((out_dir / "pv663_eval_summary.json").is_file())
            self.assertTrue((out_dir / "pv663_report.txt").is_file())
            for v in ("B0_32_Ref", "M3_Off_R", "Stage3_R_DeformGrad"):
                self.assertGreater(summary["methods"][v]["3"]["macro"]["dice"], 0.85)


if __name__ == "__main__":
    unittest.main()
