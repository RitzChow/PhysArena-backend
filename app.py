from __future__ import annotations

import json
import math
import os
import glob
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from openpyxl import load_workbook


# -------------------------
# Configuration
# -------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 定义支持的竞赛配置
COMPETITIONS_CONFIG = {
    "ipho2024": {
        "file_pattern": "ipho2024_outputs.xlsx",
        "key": "ipho--ipho_2024",
        "nice_name": "IPhO 2024",
        "index": 1
    },
    "ipho2025": {
        "file_pattern": "ipho2025_outputs.xlsx", 
        "key": "ipho--ipho_2025",
        "nice_name": "IPhO 2025",
        "index": 2
    },
    # 可以继续添加更多竞赛
    # "aime2024": {
    #     "file_pattern": "aime2024_outputs.xlsx",
    #     "key": "aime--aime_2024", 
    #     "nice_name": "AIME 2024",
    #     "index": 3
    # }
}


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


def find_competition_files() -> Dict[str, str]:
    """查找所有存在的竞赛文件"""
    found_files = {}
    for comp_id, config in COMPETITIONS_CONFIG.items():
        file_path = os.path.join(PROJECT_ROOT, config["file_pattern"])
        if os.path.exists(file_path):
            found_files[comp_id] = file_path
    return found_files


def read_outputs_xlsx(path: str) -> List[RunRecord]:
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    headers = [str(h) if h is not None else "" for h in rows[0]]
    idx = {h: i for i, h in enumerate(headers)}

    def get(row, key, default=None):
        i = idx.get(key)
        return row[i] if i is not None and i < len(row) else default

    data: List[RunRecord] = []
    for row in rows[1:]:
        if not any(x is not None for x in row):
            continue
        try:
            record = RunRecord(
                problem_idx=str(get(row, "problem_idx")),
                problem_statement=str(get(row, "problem")),
                model_name=str(get(row, "model_name")),
                model_config=str(get(row, "model_config")),
                idx_answer=int(get(row, "idx_answer", 0) or 0),
                user_message=str(get(row, "user_message")),
                answer=str(get(row, "answer")),
                messages=str(get(row, "messages")),
                input_tokens=float(get(row, "input_tokens", 0) or 0),
                output_tokens=float(get(row, "output_tokens", 0) or 0),
                run_cost=float(get(row, "cost", 0) or 0),
                input_cost_per_tokens=float(get(row, "input_cost_per_tokens", 0) or 0),
                output_cost_per_tokens=float(get(row, "output_cost_per_tokens", 0) or 0),
                gold_answer=(get(row, "gold_answer") if get(row, "gold_answer") is not None else None),
                parsed_answer=(get(row, "parsed_answer") if get(row, "parsed_answer") is not None else None),
                correct=bool(get(row, "correct")) if get(row, "correct") is not None else None,
            )
            data.append(record)
        except Exception as e:
            print(f"跳过格式错误的行: {e}")
            continue
    return data


def extract_problem_order_from_runs(runs: List[RunRecord]) -> List[str]:
    """从运行记录中提取问题ID的顺序"""
    problem_ids = []
    seen = set()
    for r in runs:
        if r.problem_idx not in seen:
            problem_ids.append(r.problem_idx)
            seen.add(r.problem_idx)
    return problem_ids


def build_competition_data(comp_id: str, file_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, bool], Dict[Tuple[str, int], Dict[str, Any]]]:
    """为单个竞赛构建数据"""
    config = COMPETITIONS_CONFIG[comp_id]
    runs = read_outputs_xlsx(file_path)
    
    if not runs:
        return {}, [], {}, {}

    # 从runs中提取问题顺序
    problem_ids_in_order = extract_problem_order_from_runs(runs)

    # Group runs: per (model_name, problem_id)
    grouped: Dict[Tuple[str, str], List[RunRecord]] = defaultdict(list)
    model_set: set[str] = set()
    for r in runs:
        model_set.add(r.model_name)
        grouped[(r.model_name, r.problem_idx)].append(r)

    # Aggregate per model totals for tokens and cost
    model_totals = {m: {"input_tokens": 0.0, "output_tokens": 0.0, "cost": 0.0} for m in model_set}
    model_price: Dict[str, Dict[str, float]] = {m: {"input": None, "output": None} for m in model_set}

    for r in runs:
        mt = model_totals[r.model_name]
        mt["input_tokens"] += r.input_tokens or 0
        mt["output_tokens"] += r.output_tokens or 0
        mt["cost"] += r.run_cost or 0
        
        if model_price[r.model_name]["input"] is None and r.input_cost_per_tokens is not None:
            v = float(r.input_cost_per_tokens or 0)
            model_price[r.model_name]["input"] = v
        if model_price[r.model_name]["output"] is None and r.output_cost_per_tokens is not None:
            v = float(r.output_cost_per_tokens or 0)
            model_price[r.model_name]["output"] = v

    # Build results rows
    numeric_to_pid: Dict[int, str] = {}
    problem_names: List[str] = []
    for i, pid in enumerate(problem_ids_in_order, start=1):
        numeric_to_pid[i] = pid
        problem_names.append(pid)

    results_rows: List[Dict[str, Any]] = []
    for idx in range(1, len(problem_ids_in_order) + 1):
        pid = numeric_to_pid[idx]
        row: Dict[str, Any] = {"question": idx}
        for m in sorted(model_set):
            runs_for = grouped.get((m, pid), [])
            if not runs_for:
                row[m] = 0
            else:
                num = sum(1 for r in runs_for if (r.correct is True))
                den = len(runs_for)
                acc = 100.0 * num / den if den > 0 else 0.0
                row[m] = acc
        results_rows.append(row)

    # Avg row
    avg_row: Dict[str, Any] = {"question": "Avg"}
    for m in sorted(model_set):
        vals = [r[m] for r in results_rows if isinstance(r[m], (int, float))]
        avg_row[m] = sum(vals) / len(vals) if vals else 0.0
    results_rows.append(avg_row)

    # Cost row
    cost_row: Dict[str, Any] = {"question": "Cost"}
    for m in sorted(model_set):
        cost_row[m] = model_totals[m]["cost"]
    results_rows.append(cost_row)

    # Build secondary data
    secondary_rows: List[Dict[str, Any]] = []
    
    # Input Tokens
    row_it = {"question": "Input Tokens"}
    for m in sorted(model_set):
        row_it[m] = model_totals[m]["input_tokens"]
    secondary_rows.append(row_it)

    # Input Cost
    row_icpt = {"question": "Input Cost"}
    for m in sorted(model_set):
        price_per_mtok = model_price[m]["input"] or 0.0
        tokens = model_totals[m]["input_tokens"] or 0.0
        dollars = (tokens * price_per_mtok) / 1_000_000.0
        row_icpt[m] = round(dollars, 6)
    secondary_rows.append(row_icpt)

    # Output Tokens
    row_ot = {"question": "Output Tokens"}
    for m in sorted(model_set):
        row_ot[m] = model_totals[m]["output_tokens"]
    secondary_rows.append(row_ot)

    # Output Cost
    row_ocpt = {"question": "Output Cost"}
    for m in sorted(model_set):
        price_per_mtok = model_price[m]["output"] or 0.0
        tokens = model_totals[m]["output_tokens"] or 0.0
        dollars = (tokens * price_per_mtok) / 1_000_000.0
        row_ocpt[m] = round(dollars, 6)
    secondary_rows.append(row_ocpt)

    # Acc
    row_acc = {"question": "Acc"}
    for m in sorted(model_set):
        row_acc[m] = avg_row[m]
    secondary_rows.append(row_acc)

    # Competition info
    competition_info = {
        "index": config["index"],
        "nice_name": config["nice_name"],
        "type": "FinalAnswer",
        "num_problems": len(problem_ids_in_order),
        "medal_thresholds": [75, 50, 25],
        "judge": False,
        "problem_names": problem_names,
    }

    # Competition dates (no contamination)
    competition_dates = {}
    for m in model_set:
        competition_dates[m] = False

    # Build traces index
    traces_index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for idx in range(1, len(problem_ids_in_order) + 1):
        pid = numeric_to_pid[idx]
        for m in model_set:
            runs_for = sorted(grouped.get((m, pid), []), key=lambda r: r.idx_answer)
            if not runs_for:
                continue
            statement = runs_for[0].problem_statement or ""
            gold = runs_for[0].gold_answer or ""
            outs: List[Dict[str, Any]] = []
            for rr in runs_for:
                outs.append({
                    "parsed_answer": rr.parsed_answer,
                    "correct": bool(rr.correct) if rr.correct is not None else False,
                    "solution": rr.answer or "",
                })
            traces_index[(m, idx)] = {
                "statement": statement,
                "gold_answer": gold,
                "model_outputs": outs,
            }

    return competition_info, results_rows, secondary_rows, competition_dates, traces_index


def build_backend_payload() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """构建所有竞赛的后端数据"""
    found_files = find_competition_files()
    
    if not found_files:
        print("警告: 未找到任何竞赛文件")
        return {}, {}, {}, {}

    all_competition_info = {}
    all_results = {}
    all_secondary = {}
    all_competition_dates = {}
    all_traces_index = {}

    for comp_id, file_path in found_files.items():
        print(f"处理竞赛文件: {comp_id} -> {file_path}")
        try:
            config = COMPETITIONS_CONFIG[comp_id]
            comp_info, results_rows, secondary_rows, comp_dates, traces_idx = build_competition_data(comp_id, file_path)
            
            comp_key = config["key"]
            all_competition_info[comp_key] = comp_info
            all_results[comp_key] = results_rows
            all_secondary[comp_key] = secondary_rows
            all_competition_dates[comp_key] = comp_dates
            
            # 转换traces_index的key格式
            for (model, task_idx), trace_data in traces_idx.items():
                all_traces_index[(comp_key, model, task_idx)] = trace_data
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue

    results_payload = {
        "competition_info": all_competition_info,
        "results": all_results
    }

    return results_payload, all_secondary, all_competition_dates, all_traces_index


# -------------------------
# Flask app
# -------------------------
app = Flask(__name__, static_folder=PROJECT_ROOT, static_url_path="")
CORS(app)

# 在应用启动时构建数据
print("正在构建后端数据...")
RESULTS_PAYLOAD, SECONDARY_PAYLOAD, COMP_DATES_PAYLOAD, TRACES_INDEX = build_backend_payload()
print(f"成功加载 {len(RESULTS_PAYLOAD.get('competition_info', {}))} 个竞赛")


@app.route("/")
def root():
    return send_from_directory(PROJECT_ROOT, "index.html")


@app.get("/results")
def get_results():
    return jsonify(RESULTS_PAYLOAD)


@app.get("/secondary")
def get_secondary():
    return jsonify(SECONDARY_PAYLOAD)


@app.get("/competition_dates")
def get_comp_dates():
    return jsonify(COMP_DATES_PAYLOAD)


@app.get("/traces/<competition>/<model>/<int:task>")
def get_traces(competition: str, model: str, task: int):
    key = (competition, model, task)
    data = TRACES_INDEX.get(key)
    if not data:
        return jsonify({"error": "trace not found"}), 404
    return jsonify(data)


def main():
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()
