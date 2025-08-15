from __future__ import annotations

import os
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from openpyxl import load_workbook

# -------------------------
DATA_DIR = Path(__file__).with_name("data")

app = Flask(__name__, static_url_path="", static_folder=str(Path(__file__).parent))
CORS(app)

# -------------------------
@dataclass
class RunRecord:
    problem_idx: str
    problem_statement: str
    model_name: str
    model_config: str
    idx_answer: int
    user_message: str
    answer: str
    messages: str
    input_tokens: float
    output_tokens: float
    run_cost: float
    input_cost_per_tokens: float
    output_cost_per_tokens: float
    gold_answer: str | None
    parsed_answer: str | None
    correct: bool | None

def read_outputs_xlsx(path: Path) -> List[RunRecord]:
    ws = load_workbook(path, read_only=True, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    hdr = {h: i for i, h in enumerate([str(c) if c is not None else "" for c in rows[0]])}
    def get(r, k, d=None): return r[hdr[k]] if k in hdr and hdr[k] < len(r) else d
    recs = []
    for r in rows[1:]:
        if not any(r): continue
        try:
            recs.append(RunRecord(
                problem_idx=str(get(r, "problem_idx")),
                problem_statement=str(get(r, "problem")),
                model_name=str(get(r, "model_name")),
                model_config=str(get(r, "model_config")),
                idx_answer=int(get(r, "idx_answer") or 0),
                user_message=str(get(r, "user_message")),
                answer=str(get(r, "answer")),
                messages=str(get(r, "messages")),
                input_tokens=float(get(r, "input_tokens") or 0),
                output_tokens=float(get(r, "output_tokens") or 0),
                run_cost=float(get(r, "cost") or 0),
                input_cost_per_tokens=float(get(r, "input_cost_per_tokens") or 0),
                output_cost_per_tokens=float(get(r, "output_cost_per_tokens") or 0),
                gold_answer=get(r, "gold_answer"),
                parsed_answer=get(r, "parsed_answer"),
                correct=bool(get(r, "correct")) if get(r, "correct") is not None else None,
            ))
        except Exception:
            continue
    return recs

# ------------------------------------------------------------------
# 一次性读取所有 xlsx
# ------------------------------------------------------------------
ALL_RESULTS: Dict[str, Any] = {}         
ALL_SECONDARY: Dict[str, Any] = {}       
ALL_DATES: Dict[str, Any] = {}            
ALL_TRACES: Dict[str, Dict[str, Any]] = {}  

def load_all():
    ALL_RESULTS.clear(); ALL_SECONDARY.clear(); ALL_DATES.clear(); ALL_TRACES.clear()
    for xlsx in DATA_DIR.glob("*_outputs.xlsx"):
        comp_key = xlsx.stem.replace("_outputs", "")
        runs = read_outputs_xlsx(xlsx)

        # 题号顺序
        pid_order: List[str] = []
        pid_gold: Dict[str, str] = {}
        seen = set()
        for r in runs:
            if r.problem_idx not in seen:
                seen.add(r.problem_idx)
                pid_order.append(r.problem_idx)
                pid_gold[r.problem_idx] = r.gold_answer or ""

        grouped = defaultdict(list)
        models = set()
        for r in runs:
            grouped[(r.model_name, r.problem_idx)].append(r)
            models.add(r.model_name)

        # 统计
        totals = {m: {"input": 0.0, "output": 0.0, "cost": 0.0} for m in models}
        price  = {m: {"in": None, "out": None} for m in models}
        for r in runs:
            totals[r.model_name]["input"]  += r.input_tokens or 0
            totals[r.model_name]["output"] += r.output_tokens or 0
            totals[r.model_name]["cost"]   += r.run_cost or 0
            if price[r.model_name]["in"]  is None: price[r.model_name]["in"]  = float(r.input_cost_per_tokens  or 0)
            if price[r.model_name]["out"] is None: price[r.model_name]["out"] = float(r.output_cost_per_tokens or 0)

        # results_rows
        rows = []
        for idx, pid in enumerate(pid_order, 1):
            row = {"question": idx}
            for m in sorted(models):
                rs = grouped.get((m, pid), [])
                num = sum(1 for r in rs if r.correct is True)
                den = len(rs)
                row[m] = 100.0 * num / den if den else 0.0
            rows.append(row)

        avg = {"question": "Avg"}
        for m in sorted(models):
            avg[m] = sum(r[m] for r in rows) / len(rows)
        rows.append(avg)

        cost_row = {"question": "Cost"}
        for m in sorted(models):
            cost_row[m] = totals[m]["cost"]
        rows.append(cost_row)

        ALL_RESULTS[comp_key] = {
            "competition_info": {
               **{comp_key: {
                    "index": 1,
                    "nice_name": comp_key.replace("_", " ").upper(),
                    "type": "FinalAnswer",
                    "num_problems": len(pid_order),
                    "medal_thresholds": [75, 50, 25],
                    "judge": False,
                    "problem_names": pid_order,
                }
            }
        },
            "results": {**{comp_key: rows}}
        }

        ALL_SECONDARY[comp_key] = {
            **{comp_key: [
                {"question": "Input Tokens",
                 **{m: totals[m]["input"] for m in sorted(models)}},
                {"question": "Input Cost",
                 **{m: round(totals[m]["input"] * price[m]["in"] / 1e6, 6) for m in sorted(models)}},
                {"question": "Output Tokens",
                 **{m: totals[m]["output"] for m in sorted(models)}},
                {"question": "Output Cost",
                 **{m: round(totals[m]["output"] * price[m]["out"] / 1e6, 6) for m in sorted(models)}},
                {"question": "Acc", **{m: avg[m] for m in sorted(models)}},
            ]
        }
        }

        ALL_DATES[comp_key] = {**{comp_key: {m: False for m in models}}}

        # traces
        traces_index = {}
        for idx, pid in enumerate(pid_order, 1):
            gold = pid_gold[pid]
            for m in sorted(models):
                rs = sorted(grouped.get((m, pid), []), key=lambda r: r.idx_answer)
                if not rs: continue
                traces_index[(m, idx)] = {
                    "statement": rs[0].problem_statement or "",
                    "gold_answer": gold,
                    "model_outputs": [
                        {"parsed_answer": r.parsed_answer,
                         "correct": bool(r.correct) if r.correct is not None else False,
                         "solution": r.answer or ""} for r in rs
                    ]
                }
        ALL_TRACES[comp_key] = traces_index

load_all()

# ------------------------------------------------------------------
# 统一路由：一次性返回所有竞赛
# ------------------------------------------------------------------
@app.route("/results")
def all_results():
    return jsonify(ALL_RESULTS)

@app.route("/secondary")
def all_secondary():
    return jsonify(ALL_SECONDARY)

@app.route("/competition_dates")
def all_dates():
    return jsonify(ALL_DATES)

@app.route("/traces/<competition>/<model>/<int:task>")
def get_trace(competition: str, model: str, task: int):
    if competition not in ALL_TRACES:
        return jsonify({"error": "competition not found"}), 404
    key = (model, task)
    data = ALL_TRACES[competition].get(key)
    if not data:
        return jsonify({"error": "trace not found"}), 404
    return jsonify(data)

# 静态首页
@app.route("/")
def root():
    return send_from_directory(Path(__file__).parent, "index.html")

# 入口
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
