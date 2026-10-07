"""scripts/draft_report.py copies numbers out of results/ -- it must never invent one,
and must not crash on a partial results/ directory (students run it between notebooks)."""
from __future__ import annotations

import csv
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("draft_report", ROOT / "scripts" / "draft_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scores(t, r, f, lat, n=50):
    return {"target": t, "regression": r, "format": f, "latency_ms": lat, "n": n, "extra": {}}


def _write_full_results(d: pathlib.Path) -> None:
    w = lambda name, obj: (d / name).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    w("mask_proof.json", {"mask_mode": "assistant-only", "n_supervised": 31, "n_total": 212,
                          "supervised_fraction": 0.1462, "answer_is_supervised": True,
                          "question_is_masked": True, "supervised_preview": '{"intent": "doi_tra"}',
                          "masked_preview": "..."})
    w("template_check.json", {"ok": False, "open_tag_present": True, "body_present": False,
                              "rendered": "...", "verdict": "template STRIPS reasoning"})
    w("token_stats.json", {"n": 250, "mean": 190.2, "p95": 231, "max": 260, "suggested_max_length": 256})
    w("baselines_frozen.json", {"tier": "T4", "model": "Qwen/Qwen3.5-4B",
                                "baseline_a": _scores(0.412, 0.70, 0.36, 2100.0),
                                "baseline_b": _scores(0.655, 0.71, 0.98, 1566.0),
                                "optimized_prompt_sha": "abc123", "n_target": 50, "n_regression": 15,
                                "eval_limit": None, "smoke_mode": False})
    w("verdict.json", {"comparison": [
        {"run": "(a) base + naive prompt", "target": 0.412, "regression": 0.70, "format": 0.36, "latency_ms": 2100.0, "n": 50},
        {"run": "(b) base + optimized prompt", "target": 0.655, "regression": 0.71, "format": 0.98, "latency_ms": 1566.0, "n": 50},
        {"run": "(c) LoRA fine-tune", "target": 0.871, "regression": 0.66, "format": 1.0, "latency_ms": 900.5, "n": 50}],
        "verdict": {"passed": False, "reasons": ["general capability regressed by 0.050"],
                    "target_delta": 0.216, "regression_delta": -0.05},
        "valid_trace_rate": 0.0})
    w("autopsy.json", [{"run": "correct", "target": 0.871, "format": 1.0, "latency_ms": 900.5, "n": 50},
                       {"run": "attn_only", "target": 0.88, "format": 1.0, "latency_ms": 899.0, "n": 50},
                       {"run": "wrong_lr", "target": 0.52, "format": 0.9, "latency_ms": 950.0, "n": 50},
                       {"run": "qlora", "target": 0.80, "format": 1.0, "latency_ms": 1500.0, "n": 50}])
    w("qualitative.json", [{"i": i, "ticket": f"ticket | {i}", "ft_score": s, "ft_pred": "{}"}
                           for i, s in enumerate([1.0, 0.25, 0.5, 1.0, 0.75, 0.0])])
    cols = ["run", "placement", "r", "trainable_params", "learning_rate", "final_loss",
            "max_steps", "train_seconds", "peak_vram_gb", "mask_mode", "tier", "model", "precision"]
    rows = [["correct", "text-linear", 16, 10000000, 2e-4, 0.31, 30, 1021.0, 11.2, "assistant-only", "T4", "Qwen/Qwen3.5-4B", "fp16"],
            ["attn_only", "q,v", 283, 10100000, 2e-4, 0.29, 30, 900.0, 11.0, "", "T4", "Qwen/Qwen3.5-4B", "fp16"],
            ["wrong_lr", "text-linear", 16, 10000000, 2e-5, 0.90, 30, 1000.0, 11.2, "", "T4", "Qwen/Qwen3.5-4B", "fp16"],
            ["qlora", "text-linear", 16, 10000000, 2e-4, 0.30, 30, 1400.0, 6.1, "", "T4", "Qwen/Qwen3.5-4B", "fp16"]]
    with (d / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        wr.writerows(rows)


def test_empty_results_reports_missing_without_inventing_numbers(tmp_path):
    text, missing = _load().build(tmp_path)
    assert "mask_proof.json (NB1)" in missing and "verdict.json (NB5)" in missing
    assert "PASSED" not in text and "FAILED" not in text


def test_full_results_numbers_are_copied_verbatim(tmp_path):
    _write_full_results(tmp_path)
    text, missing = _load().build(tmp_path)
    assert missing == []
    assert "0.1462" in text                              # mask proof
    assert "0.655" in text and "0.871" in text           # baseline (b), fine-tune
    assert "`FAILED`" in text and "-0.050" in text       # verdict + regression delta
    assert "1.00%" in text                               # attn_only budget drift
    assert "Hai thứ tự **KHÁC** nhau" in text            # loss order != target order
    assert "5.10 GB" in text                             # qlora VRAM saving
    assert "ticket \\| 1" in text                        # pipes escaped in table cells
