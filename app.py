from __future__ import annotations

import os
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from openpyxl import load_workbook

# -------------------------
# 配置
# -------------------------
DATA_DIR = Path(__file__).with_name("data")
app = Flask(__name__, static_url_path="", static_folder=str(Path(__file__).parent))
CORS(app)

# -------------------------
# 数据类定义
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

# -------------------------
# 全局变量：用于缓存加载后的数据
# -------------------------
# 修改：使用一个统一的 payload 存储所有数据
ALL_DATA_PAYLOAD: Dict[str, Any] = {}
ALL_TRACES: Dict[str, Dict[str, Any]] = {}  # Traces 结构保持不变

# -------------------------
# 数据读取与处理函数
# -------------------------
def read_outputs_xlsx(path: Path) -> List[RunRecord]:
    ws = load_workbook(path, read_only=True, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    if not rows: return []
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
        except (ValueError, IndexError, KeyError) as e:
            print(f"Skipping row due to error in file {path.name}: {e}")
            continue
    return recs

def load_all():
    """
    一次性读取所有xlsx文件, 并将数据处理成前端期望的聚合格式。
    """
    global ALL_DATA_PAYLOAD, ALL_TRACES
    
    # 用于聚合所有比赛数据的临时字典
    final_results = {}
    final_competition_info = {}
    final_secondary = {}
    final_dates = {}
    
    ALL_TRACES.clear()

    # 遍历data文件夹下的所有outputs.xlsx文件
    for xlsx_file_index, xlsx in enumerate(sorted(DATA_DIR.glob("*_outputs.xlsx"))):
        comp_key = xlsx.stem.replace("_outputs", "")
        runs = read_outputs_xlsx(xlsx)

        if not runs:
            print(f"Warning: No data found in {xlsx.name}, skipping.")
            continue

        # 题号顺序与正确答案
        pid_order: List[str] = []
        pid_gold: Dict[str, str] = {}
        seen_pids = set()
        for r in runs:
            if r.problem_idx not in seen_pids:
                seen_pids.add(r.problem_idx)
                pid_order.append(r.problem_idx)
                pid_gold[r.problem_idx] = r.gold_answer or ""

        # 按 (模型, 题号) 分组
        grouped = defaultdict(list)
        models = sorted(list({r.model_name for r in runs}))
        for r in runs:
            grouped[(r.model_name, r.problem_idx)].append(r)

        # 统计 token 和 cost
        totals = {m: {"input": 0.0, "output": 0.0, "cost": 0.0} for m in models}
        price = {m: {"in": 0.0, "out": 0.0} for m in models}
        for r in runs:
            totals[r.model_name]["input"]  += r.input_tokens
            totals[r.model_name]["output"] += r.output_tokens
            totals[r.model_name]["cost"]   += r.run_cost
            if price[r.model_name]["in"] == 0.0: price[r.model_name]["in"] = r.input_cost_per_tokens
            if price[r.model_name]["out"] == 0.0: price[r.model_name]["out"] = r.output_cost_per_tokens

        # 构建主表格 (results_rows)
        rows = []
        for idx, pid in enumerate(pid_order, 1):
            row = {"question": idx}
            for m in models:
                rs = grouped.get((m, pid), [])
                num = sum(1 for r in rs if r.correct is True)
                den = len(rs)
                row[m] = 100.0 * num / den if den else 0.0
            rows.append(row)

        avg = {"question": "Avg"}
        for m in models:
            avg[m] = sum(r[m] for r in rows) / len(rows) if rows else 0.0
        rows.append(avg)

        cost_row = {"question": "Cost"}
        for m in models:
            cost_row[m] = totals[m]["cost"]
        rows.append(cost_row)

        # --- 数据聚合 ---
        # 将当前比赛的数据添加到 final 字典中
        final_results[comp_key] = rows
        final_competition_info[comp_key] = {
            "index": xlsx_file_index,  # 使用文件顺序作为索引
            "nice_name": comp_key.replace("_", " ").title(),
            "type": "FinalAnswer",
            "num_problems": len(pid_order),
            "medal_thresholds": [75, 50, 25],
            "judge": False,
            "problem_names": pid_order,
        }
        final_secondary[comp_key] = [
            {"question": "Input Tokens", **{m: totals[m]["input"] for m in models}},
            {"question": "Input Cost", **{m: round(totals[m]["input"] * price[m]["in"] / 1e6, 6) if price[m]["in"] else 0 for m in models}},
            {"question": "Output Tokens", **{m: totals[m]["output"] for m in models}},
            {"question": "Output Cost", **{m: round(totals[m]["output"] * price[m]["out"] / 1e6, 6) if price[m]["out"] else 0 for m in models}},
            {"question": "Acc", **{m: avg.get(m, 0.0) for m in models}},
        ]
        final_dates[comp_key] = {m: False for m in models}

        # 处理 Traces
        traces_index = {}
        for idx, pid in enumerate(pid_order, 1):
            gold = pid_gold[pid]
            for m in models:
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

    # --- 构建最终 Payload ---
    # 循环结束后，将聚合好的数据组装成一个大的字典
    ALL_DATA_PAYLOAD = {
        "results": final_results,
        "competition_info": final_competition_info,
        "secondary": final_secondary,
        "competition_dates": final_dates,
    }

# ------------------------------------------------------------------
# Flask API 路由
# ------------------------------------------------------------------

# 修改：创建一个统一的路由返回所有启动所需的数据
@app.route("/api/all_data")
def all_data():
    """返回所有聚合后的数据，供前端一次性加载。"""
    return jsonify(ALL_DATA_PAYLOAD)

# 保持不变：用于获取单个题目的详细信息
@app.route("/traces/<competition>/<model>/<int:task>")
def get_trace(competition: str, model: str, task: int):
    if competition not in ALL_TRACES:
        return jsonify({"error": "competition not found"}), 404
    
    # 在 Python 中，元组 (tuple) 用作字典键是常见的
    key = (model, task)
    data = ALL_TRACES[competition].get(key)
    
    if not data:
        return jsonify({"error": "trace not found"}), 404
    return jsonify(data)

# 保持不变：提供静态首页
@app.route("/")
def root():
    return send_from_directory(Path(__file__).parent, "index.html")

# ------------------------------------------------------------------
# 程序入口
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 在启动服务器前先加载所有数据
    print("Loading all data from xlsx files...")
    load_all()
    print("Data loaded successfully.")
    
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False) # 在生产环境中建议关闭 debug
