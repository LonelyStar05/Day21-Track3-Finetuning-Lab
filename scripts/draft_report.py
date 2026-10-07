"""Draft submission/REPORT_DRAFT.md from whatever is in results/.

Every number in the draft is copied from a results/ file, never typed by hand, so the
tables cannot drift from what the grader cross-checks (rubric 4.3). The prose -- why
you chose the model, what the verdict means, what you learned -- is left as `<viết>`
placeholders: that part is graded on being yours.

Missing files are reported, not invented. Run it after each notebook if you like; it
fills in whatever exists so far.

    python scripts/draft_report.py              # -> submission/REPORT_DRAFT.md
    python scripts/draft_report.py --stdout     # print instead of writing
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
TODO = "`<viết>`"
MISSING = "_(chưa có — chạy {nb} trước)_"


def _json(results: pathlib.Path, name: str):
    p = results / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _runs(results: pathlib.Path) -> dict[str, dict]:
    """Last row per run key: rows are appended, so a retrain supersedes the old row."""
    p = results / "runs.csv"
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    with p.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("run"):
                out[row["run"]] = row
    return out


def _count_lines(path: pathlib.Path) -> int | None:
    if not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _fmt(v, nd: int = 3) -> str:
    if v is None or v == "":
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f.is_integer() and abs(f) >= 100 else f"{f:.{nd}f}"


def _table(cols: list[str], rows: list[list]) -> str:
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    return "\n".join([head, rule, *("| " + " | ".join(str(c) for c in r) + " |" for r in rows)])


def section_setup(results, missing) -> str:
    stats = _json(results, "token_stats.json")
    tmpl = _json(results, "template_check.json")
    frozen = _json(results, "baselines_frozen.json") or {}
    runs = _runs(results)
    correct = runs.get("correct", {})
    n_train = _count_lines(ROOT / "data" / "split" / "train.jsonl")
    n_val = _count_lines(ROOT / "data" / "split" / "val.jsonl")
    if stats is None:
        missing.append("token_stats.json (NB1)")
    if tmpl is None:
        missing.append("template_check.json (NB1)")

    p95 = stats.get("p95", stats.get("suggested_max_length")) if stats else None
    lines = [
        "## 1. Setup",
        "",
        "| | |",
        "|---|---|",
        f"| Tier | `{frozen.get('tier') or correct.get('tier') or '—'}` |",
        f"| Base model | `{frozen.get('model') or correct.get('model') or '—'}` |",
        f"| Precision | `{correct.get('precision') or '—'}` |",
        f"| Train / val | {n_train if n_train is not None else '—'} / "
        f"{n_val if n_val is not None else '—'} (seed 42) |",
        f"| `max_length` gợi ý (p95 → luỹ thừa 2) | "
        f"{stats.get('suggested_max_length', '—') if stats else '—'} *(results/token_stats.json)* |",
        f"| Token stats | " + (", ".join(f"{k}={v}" for k, v in stats.items()) if stats else "—") + " |",
        f"| `MASK_MODE` | `{correct.get('mask_mode') or '—'}` |",
        f"| `max_steps` | {correct.get('max_steps') or '—'} |",
        "",
        "**Model + dataset đã chọn và lý do (rubric 4.1):** " + TODO,
        "",
    ]
    if tmpl is not None:
        lines += [
            f"**Template có giữ khối `<think>` không?** "
            f"{'có' if tmpl.get('body_present') else 'không'} — "
            f"verdict: _{tmpl.get('verdict')}_ *(results/template_check.json)*",
            "",
            "Nếu không: bạn đã xử lý thế nào? " + TODO,
            "",
        ]
    if p95 is not None and stats:
        lines += ["Nếu `max_length` của tier khác số gợi ý ở trên, giải thích vì sao (rubric 1.3): " + TODO, ""]
    return "\n".join(lines)


def section_mask(results, missing) -> str:
    proof = _json(results, "mask_proof.json")
    if proof is None:
        missing.append("mask_proof.json (NB1)")
        return "## 2. Mask proof (NB1)\n\n" + MISSING.format(nb="NB1") + "\n"
    rows = [
        ["`mask_mode`", f"`{proof.get('mask_mode')}`"],
        ["`supervised_fraction`", proof.get("supervised_fraction")],
        ["`n_supervised` / `n_total`", f"{proof.get('n_supervised')} / {proof.get('n_total')}"],
        ["Câu trả lời nằm trong loss", f"`{str(proof.get('answer_is_supervised')).lower()}`"],
        ["Câu hỏi KHÔNG nằm trong loss", f"`{str(proof.get('question_is_masked')).lower()}`"],
    ]
    warn = ""
    if float(proof.get("supervised_fraction") or 0) >= 0.95:
        warn = "\n> ⚠ `supervised_fraction ≥ 0.95` — loss đang tính cả prompt. Mất trắng rubric 1.1.\n"
    return "\n".join([
        "## 2. Mask proof (NB1)", "", _table(["", ""], rows), warn,
        "Đoạn được tính loss (`supervised_preview`):", "", "```",
        str(proof.get("supervised_preview", "")).strip(), "```", "",
    ])


def section_baselines(results, missing) -> str:
    frozen = _json(results, "baselines_frozen.json")
    verdict = _json(results, "verdict.json")
    if frozen is None:
        missing.append("baselines_frozen.json (NB2)")
        return "## 3. Ba baseline (NB2 — đo TRƯỚC khi train)\n\n" + MISSING.format(nb="NB2") + "\n"
    rows = []
    for name, key in (("(a) base + naive prompt", "baseline_a"), ("(b) base + optimized prompt", "baseline_b")):
        s = frozen[key]
        rows.append([name, _fmt(s["target"]), _fmt(s["regression"]), _fmt(s["format"]),
                     _fmt(s["latency_ms"], 1), s.get("n")])
    if verdict is not None:
        for r in verdict.get("comparison", []):
            if r["run"].startswith("(c)"):
                rows.append([r["run"], _fmt(r["target"]), _fmt(r["regression"]), _fmt(r["format"]),
                             _fmt(r["latency_ms"], 1), r.get("n")])
    else:
        rows.append(["(c) LoRA fine-tune", "—", "—", "—", "—", "_(NB5)_"])
    a, b = frozen["baseline_a"]["target"], frozen["baseline_b"]["target"]
    smoke = ""
    if frozen.get("smoke_mode"):
        smoke = (f"\n> ⚠ Chạy ở chế độ rút gọn `EVAL_LIMIT={frozen.get('eval_limit')}`. "
                 "Bài nộp phải chạy mặc định.\n")
    return "\n".join([
        "## 3. Ba baseline (NB2 — đo TRƯỚC khi train)", "",
        _table(["Run", "target", "regression", "format", "latency (ms)", "n"], rows), smoke,
        f"SHA prompt (b): `{frozen.get('optimized_prompt_sha')}` · "
        f"n_target={frozen.get('n_target')} · n_regression={frozen.get('n_regression')}",
        "",
        f"**(b) có thật sự mạnh hơn (a) không?** {'có' if b > a else '**không**'} "
        f"(target {b:.3f} vs {a:.3f}, Δ {b - a:+.3f}).",
        "Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: làm mạnh lên hay yếu đi, và vì sao? " + TODO,
        "",
    ])


def section_autopsy(results, missing) -> str:
    runs = _runs(results)
    autopsy = {r["run"]: r for r in (_json(results, "autopsy.json") or [])}
    if not runs:
        missing.append("runs.csv (NB3/NB4)")
    if not autopsy:
        missing.append("autopsy.json (NB5)")
    rows = []
    for key in ("correct", "attn_only", "wrong_lr", "qlora"):
        r, a = runs.get(key, {}), autopsy.get(key, {})
        rows.append([f"`{key}`", r.get("placement", "—"), r.get("r", "—"), r.get("trainable_params", "—"),
                     r.get("learning_rate", "—"), _fmt(r.get("final_loss"), 4),
                     f"**{_fmt(a.get('target'))}**" if a else "—", _fmt(a.get("format")) if a else "—",
                     r.get("max_steps", "—"), _fmt(r.get("train_seconds"), 1), _fmt(r.get("peak_vram_gb"), 2)])

    notes = []
    c, att = runs.get("correct"), runs.get("attn_only")
    if c and att and c.get("trainable_params") and att.get("trainable_params"):
        tc, ta = int(c["trainable_params"]), int(att["trainable_params"])
        notes.append(f"Sai lệch ngân sách tham số `attn_only` vs `correct`: {abs(ta - tc) / tc:.2%} (phải < 5%).")
    steps = {runs[k].get("max_steps") for k in runs if k in ("correct", "attn_only", "wrong_lr", "qlora")}
    if len(steps) > 1:
        notes.append(f"⚠ `max_steps` không đồng nhất giữa các run: {sorted(steps)}.")
    if autopsy and runs:
        by_loss = sorted((k for k in autopsy if runs.get(k, {}).get("final_loss")),
                         key=lambda k: float(runs[k]["final_loss"]))
        by_target = sorted(autopsy, key=lambda k: -float(autopsy[k]["target"]))
        notes.append(f"Thứ tự theo target (NB5 §4): {' > '.join(by_target)}.")
        notes.append(f"Thứ tự theo train loss (NB4, thấp → cao): {' > '.join(by_loss)}.")
        notes.append("Hai thứ tự **" + ("giống" if by_loss == by_target[:len(by_loss)] else "KHÁC") + "** nhau.")
    if runs.get("correct", {}).get("peak_vram_gb") and runs.get("qlora", {}).get("peak_vram_gb"):
        saved = float(runs["correct"]["peak_vram_gb"]) - float(runs["qlora"]["peak_vram_gb"])
        notes.append(f"`qlora` tiết kiệm {saved:.2f} GB peak VRAM so với `correct`.")

    return "\n".join([
        "## 4. Giải phẫu cấu hình sai (NB4, chấm ở NB5 §4)", "",
        _table(["Run", "vị trí", "r", "trainable", "LR", "train loss (NB4)", "**target (NB5 §4)**",
                "format", "max_steps", "s", "VRAM GB"], rows), "",
        *(f"- {n}" for n in notes), "",
        "Mỗi run đổi đúng một biến so với `correct` (rubric 2.3): " + TODO, "",
        "**4.1 — rank vs vị trí gắn adapter (≥3 câu):** " + TODO, "",
        "**4.2 — `wrong_lr`: đường loss khác ra sao, nhìn loss thôi sẽ kết luận sai gì (≥3 câu):** " + TODO, "",
        "**4.3 — `qlora`: tiết kiệm bao nhiêu VRAM, trả giá bằng gì (≥3 câu):** " + TODO, "",
    ])


def section_verdict(results, missing) -> str:
    v = _json(results, "verdict.json")
    if v is None:
        missing.append("verdict.json (NB5)")
        return "## 5. Phán quyết (NB5)\n\n" + MISSING.format(nb="NB5") + "\n"
    vd = v["verdict"]
    return "\n".join([
        "## 5. Phán quyết (NB5)", "",
        f"**Kết quả cổng hồi quy**: `{'PASSED' if vd['passed'] else 'FAILED'}`",
        f"`target Δ = {vd['target_delta']:+.3f}` · `regression Δ = {vd['regression_delta']:+.3f}` · "
        f"`valid_trace_rate = {_fmt(v.get('valid_trace_rate'), 2)}`", "",
        *(f"- {r}" for r in vd.get("reasons", [])), "",
        "Diễn giải (≥100 từ; nếu FAILED: vì sao, và điều đó nói gì về bài toán): " + TODO, "",
    ])


def section_qualitative(results, missing) -> str:
    q = _json(results, "qualitative.json")
    if q is None:
        missing.append("qualitative.json (NB5)")
        return "## 6. Định tính — bắt buộc có cả ca THUA\n\n" + MISSING.format(nb="NB5") + "\n"
    q = sorted(q, key=lambda r: r["ft_score"])
    picks = q[:3] + [r for r in q[-2:] if r not in q[:3]]
    target = {}
    tpath = ROOT / "data" / "eval_target.jsonl"
    if tpath.exists():
        target = {i: json.loads(l) for i, l in enumerate(tpath.read_text(encoding="utf-8").splitlines()) if l.strip()}
    rows = []
    for n, r in enumerate(picks, 1):
        label = target.get(r["i"], {}).get("label", "—")
        label = json.dumps(label, ensure_ascii=False) if isinstance(label, dict) else label
        cell = lambda s: str(s).replace("|", "\\|")
        kind = "điểm thấp nhất" if n <= 3 else "điểm cao nhất"
        rows.append([n, r["i"], cell(r["ticket"]), cell(label), cell(r["ft_pred"]), r["ft_score"], kind])
    return "\n".join([
        "## 6. Định tính — bắt buộc có cả ca THUA", "",
        "3 ca điểm thấp nhất + 2 ca cao nhất của bản fine-tune, từ `results/qualitative.json`. "
        "File này **không** chứa dự đoán (b), nên script không tự gắn nhãn thắng/thua: lấy dự "
        "đoán (b) cho từng ca từ log NB2, thêm cột (b), rồi đánh dấu ≥2 ca **FT thua (b)** "
        "(rubric 3.4).", "",
        _table(["#", "i", "Ticket (rút gọn)", "Nhãn đúng", "(c) fine-tune", "điểm FT", "nhóm"], rows), "",
        "Có mẫu chung nào ở các ca FT thua không? " + TODO, "",
    ])


def section_merge(results) -> str:
    m = _json(results, "merge_check.json")
    if m is None:
        return ""
    return "\n".join(["## Phụ lục — NB6 merge check", "", "```json",
                      json.dumps(m, ensure_ascii=False, indent=2), "```", ""])


def build(results: pathlib.Path) -> tuple[str, list[str]]:
    missing: list[str] = []
    parts = [
        "# Lab 21 — Evaluation Report (bản nháp sinh từ results/)", "",
        "**Họ tên**: " + TODO + "  **MSSV**: " + TODO + "  **Ngày**: " + TODO, "",
        "> Bảng số được sinh tự động bởi `scripts/draft_report.py` từ `results/` — không "
        "sửa tay, chạy lại script nếu chạy lại notebook. Các chỗ " + TODO + " là phần bạn "
        "phải tự viết. Khi xong, đổi tên thành `submission/REPORT.md`.", "",
        "---", "",
        section_setup(results, missing),
        section_mask(results, missing),
        section_baselines(results, missing),
        section_autopsy(results, missing),
        section_verdict(results, missing),
        section_qualitative(results, missing),
        "## 7. Kết luận & điều tôi học được", "",
        "**Kết luận (≥150 từ, có lập luận nhân quả):** " + TODO, "",
        "**Ba điều tôi học được** (cụ thể, không generic):", "", "1. " + TODO, "2. " + TODO, "3. " + TODO, "",
        "**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** " + TODO, "",
        section_merge(results),
    ]
    if missing:
        parts.insert(6, "> **Chưa có:** " + ", ".join(missing) + "\n")
    return "\n".join(p for p in parts if p is not None), missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=pathlib.Path, default=ROOT / "results")
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "submission" / "REPORT_DRAFT.md")
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    text, missing = build(args.results)
    if args.stdout:
        print(text)
    else:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    if missing:
        print("missing:", ", ".join(missing))


if __name__ == "__main__":
    main()
