#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE = Path("/egr/research-pac/huang248/outputs/pr_diffusion/b19_solver")

POLICY_CSV = BASE / "B19_13_policy_replay_per_image.csv"
SYM_CSV = BASE / "B19_13D_flip_symmetry_candidate_audit.csv"


def parse_runs(x) -> list[int]:
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    return [int(v) for v in s.split(",") if v.strip()]


def classify(row: pd.Series) -> str:
    if float(row["psnr_identity_recomputed"]) >= 25.0:
        return "unaligned_good"
    if int(row["symmetry_rescuable25"]) == 1:
        return "symmetry_rescuable"
    return "true_bad"


def main() -> None:
    policy = pd.read_csv(POLICY_CSV).copy()
    sym = pd.read_csv(SYM_CSV).copy()

    sym["run_index"] = sym["run_index"].astype(int)
    sym["candidate_class"] = sym.apply(classify, axis=1)

    key_cols = [
        "image_id", "run_index", "candidate_class",
        "psnr_identity_recomputed", "best_aligned_psnr",
        "best_alignment", "aligned_gain",
        "exact_operator_loss",
        "bad25_unaligned", "symmetry_rescuable25", "rot180_rescuable25",
    ]

    sym_key = sym[key_cols].copy()

    rows = []

    for _, prow in policy.iterrows():
        image_id = str(prow["image_id"])
        selected_run = int(prow["selected_run"])
        candidate_runs = parse_runs(prow.get("candidate_runs", ""))
        kept_runs = parse_runs(prow.get("kept_runs", ""))

        img_sym = sym_key[sym_key["image_id"].astype(str) == image_id].copy()

        selected = img_sym[img_sym["run_index"] == selected_run]
        if selected.empty:
            print("[missing selected classification]", image_id, selected_run)
            continue
        selected = selected.iloc[0]

        cand = img_sym[img_sym["run_index"].isin(candidate_runs)].copy()
        kept = img_sym[img_sym["run_index"].isin(kept_runs)].copy()

        def count_class(df: pd.DataFrame, cls: str) -> int:
            if df.empty:
                return 0
            return int((df["candidate_class"] == cls).sum())

        rows.append({
            "image_id": image_id,
            "policy": prow["policy"],
            "policy_type": prow["policy_type"],
            "cost_full_equiv": float(prow["cost_full_equiv"]),
            "K": int(prow["K"]),
            "checkpoint_step": int(prow["checkpoint_step"]),
            "keep_k": int(prow["keep_k"]),

            "candidate_runs": ",".join(map(str, candidate_runs)),
            "kept_runs": ",".join(map(str, kept_runs)),
            "selected_run": selected_run,

            "selected_psnr": float(prow["selected_psnr"]),
            "selected_class": selected["candidate_class"],
            "selected_identity_psnr": float(selected["psnr_identity_recomputed"]),
            "selected_best_aligned_psnr": float(selected["best_aligned_psnr"]),
            "selected_best_alignment": selected["best_alignment"],
            "selected_aligned_gain": float(selected["aligned_gain"]),
            "selected_symmetry_rescuable": int(selected["symmetry_rescuable25"]),
            "selected_true_bad": int(selected["candidate_class"] == "true_bad"),
            "selected_unaligned_good": int(selected["candidate_class"] == "unaligned_good"),

            "candidate_num_unaligned_good": count_class(cand, "unaligned_good"),
            "candidate_num_symmetry_rescuable": count_class(cand, "symmetry_rescuable"),
            "candidate_num_true_bad": count_class(cand, "true_bad"),

            "kept_num_unaligned_good": count_class(kept, "unaligned_good"),
            "kept_num_symmetry_rescuable": count_class(kept, "symmetry_rescuable"),
            "kept_num_true_bad": count_class(kept, "true_bad"),

            "candidate_has_unaligned_good": int(count_class(cand, "unaligned_good") > 0),
            "candidate_has_symmetry_rescuable": int(count_class(cand, "symmetry_rescuable") > 0),
            "candidate_has_true_bad": int(count_class(cand, "true_bad") > 0),

            "kept_has_unaligned_good": int(count_class(kept, "unaligned_good") > 0),
            "kept_has_symmetry_rescuable": int(count_class(kept, "symmetry_rescuable") > 0),
            "kept_has_true_bad": int(count_class(kept, "true_bad") > 0),

            "policy_selected_bad25": int(prow["selected_bad25"]),
            "gap_to_policy_oracle": float(prow["gap_to_policy_oracle"]),
            "gap_to_oracle16": float(prow["gap_to_oracle16"]),
            "init_failure_no_good_firstK": int(prow["init_failure_no_good_firstK"]),
            "prefix_selection_failure": int(prow["prefix_selection_failure"]),
            "final_exact_selection_failure": int(prow["final_exact_selection_failure"]),
        })

    out = pd.DataFrame(rows).sort_values(["cost_full_equiv", "policy", "image_id"])

    per_image_dest = BASE / "B19_13E_policy_symmetry_classes_per_image.csv"
    out.to_csv(per_image_dest, index=False)
    print("[write]", per_image_dest)
    print("rows:", len(out))

    summary_rows = []

    for policy_name, g in out.groupby("policy"):
        g = g.copy()
        images = len(g)

        summary_rows.append({
            "policy": policy_name,
            "policy_type": g["policy_type"].iloc[0],
            "cost_full_equiv": float(g["cost_full_equiv"].iloc[0]),
            "K": int(g["K"].iloc[0]),
            "checkpoint_step": int(g["checkpoint_step"].iloc[0]),
            "keep_k": int(g["keep_k"].iloc[0]),
            "images": images,

            "selected_unaligned_good": int((g["selected_class"] == "unaligned_good").sum()),
            "selected_symmetry_rescuable": int((g["selected_class"] == "symmetry_rescuable").sum()),
            "selected_true_bad": int((g["selected_class"] == "true_bad").sum()),

            "selected_bad25": int(g["policy_selected_bad25"].sum()),
            "selected_good_after_alignment": int((g["selected_best_aligned_psnr"] >= 25.0).sum()),

            "candidate_has_symmetry_rescuable": int(g["candidate_has_symmetry_rescuable"].sum()),
            "kept_has_symmetry_rescuable": int(g["kept_has_symmetry_rescuable"].sum()),

            "total_candidate_symmetry_rescuable": int(g["candidate_num_symmetry_rescuable"].sum()),
            "total_kept_symmetry_rescuable": int(g["kept_num_symmetry_rescuable"].sum()),

            "total_candidate_true_bad": int(g["candidate_num_true_bad"].sum()),
            "total_kept_true_bad": int(g["kept_num_true_bad"].sum()),

            "kept_has_unaligned_good": int(g["kept_has_unaligned_good"].sum()),
            "mean_selected_identity_psnr": float(g["selected_identity_psnr"].mean()),
            "mean_selected_best_aligned_psnr": float(g["selected_best_aligned_psnr"].mean()),
            "min_selected_identity_psnr": float(g["selected_identity_psnr"].min()),
            "min_selected_best_aligned_psnr": float(g["selected_best_aligned_psnr"].min()),
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["cost_full_equiv", "selected_true_bad", "selected_symmetry_rescuable", "selected_unaligned_good"],
        ascending=[True, True, True, False],
    )

    summary_dest = BASE / "B19_13E_policy_symmetry_classes_summary.csv"
    summary.to_csv(summary_dest, index=False)
    print("[write]", summary_dest)

    print("\nSummary:")
    print(summary.to_string(index=False))

    print("\nSelected symmetry-rescuable cases:")
    ss = out[out["selected_class"] == "symmetry_rescuable"].copy()
    if len(ss):
        print(
            ss[[
                "policy", "image_id", "selected_run", "selected_identity_psnr",
                "selected_best_aligned_psnr", "selected_best_alignment",
                "selected_aligned_gain", "cost_full_equiv",
            ]]
            .sort_values(["policy", "image_id"])
            .to_string(index=False)
        )
    else:
        print("none")

    print("\nSelected true-bad cases:")
    tb = out[out["selected_class"] == "true_bad"].copy()
    if len(tb):
        print(
            tb[[
                "policy", "image_id", "selected_run", "selected_identity_psnr",
                "selected_best_aligned_psnr", "selected_best_alignment",
                "cost_full_equiv",
                "init_failure_no_good_firstK",
                "prefix_selection_failure",
                "final_exact_selection_failure",
            ]]
            .sort_values(["policy", "image_id"])
            .to_string(index=False)
        )
    else:
        print("none")

    print("\nPolicies that kept symmetry-rescuable candidates:")
    ks = out[out["kept_has_symmetry_rescuable"] == 1].copy()
    if len(ks):
        print(
            ks[[
                "policy", "image_id", "candidate_runs", "kept_runs",
                "kept_num_unaligned_good", "kept_num_symmetry_rescuable",
                "kept_num_true_bad", "selected_run", "selected_class",
            ]]
            .sort_values(["policy", "image_id"])
            .to_string(index=False)
        )
    else:
        print("none")


if __name__ == "__main__":
    main()
