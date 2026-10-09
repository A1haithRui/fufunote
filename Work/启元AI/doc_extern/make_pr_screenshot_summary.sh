#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="${RESULT_DIR:-/root/autodl-tmp/bench-results-official-8b}"
PARAM_DIR="${PARAM_DIR:-/root/autodl-tmp/bench-results-param-scan}"
REPORT="${REPORT:-/root/InfiniLM/doc_extern/T2-1-2_8B_服务能力优化报告.md}"
PORT="${PORT:-8102}"
PYTHON_BIN="${PYTHON_BIN:-/root/InfiniLM/.venv/bin/python}"

echo "==================== PR SCREENSHOT SUMMARY ===================="
date "+time: %Y-%m-%d %H:%M:%S %Z"
echo

echo "==================== DEVICE / SERVICE ===================="
nvidia-smi -L
nvidia-smi | sed -n '1,15p'
echo -n "health: "
"${PYTHON_BIN}" - "${PORT}" <<'PY' || true
import sys
import urllib.request

port = sys.argv[1]
with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
    print(response.read().decode("utf-8"), end="")
PY
echo
echo

echo "==================== 8B BASELINE / BEFORE ===================="
cat <<'BASELINE'
case                 con input output success out_tok/s total_tok/s mean/p99_TTFT       mean/p99_TPOT mean/p99_ITL status
small-stability       1    32    256   20/20     56.86       63.77 32.04/35.86        17.53/17.59  17.46/18.12  ok
long-prefill-fail     4  4096   1024  failed         -           - timeout/unhealthy   -            -            unhealthy
phasecap-before      16   256    256   80/80    407.30      814.65 437.84/619.32      37.70/39.20  37.54/39.61  ok
phasecap-before      64   256    256 128/128    376.64      753.37 21498.23/69212.98  51.69/53.70  51.50/61.84  ok
BASELINE

echo
echo "==================== 8B OPTIMIZED / AFTER ===================="
"${PYTHON_BIN}" - "${RESULT_DIR}" "${PARAM_DIR}" "${REPORT}" <<'PY'
import glob
import json
import os
import re
import sys

result_dir, param_dir, report_path = sys.argv[1], sys.argv[2], sys.argv[3]

def parse_log(path):
    text = open(path, errors="ignore").read()
    def grab(label):
        m = re.search(re.escape(label) + r":\s+([0-9.]+)", text)
        return m.group(1) if m else "-"
    return {
        "success": grab("Successful requests"),
        "out": grab("Output token throughput (tok/s)"),
        "total": grab("Total Token throughput (tok/s)"),
        "ttft": grab("Mean TTFT (ms)"),
        "p99ttft": grab("P99 TTFT (ms)"),
        "tpot": grab("Mean TPOT (ms)"),
        "p99tpot": grab("P99 TPOT (ms)"),
        "itl": grab("Mean ITL (ms)"),
        "p99itl": grab("P99 ITL (ms)"),
    }

def parse_json(path):
    d = json.load(open(path))
    return {
        "success": str(d["num_prompts"]),
        "out": f'{d["output_throughput"]:.2f}',
        "total": f'{d["total_token_throughput"]:.2f}',
        "ttft": f'{d["mean_ttft_ms"]:.2f}',
        "p99ttft": f'{d["p99_ttft_ms"]:.2f}',
        "tpot": f'{d["mean_tpot_ms"]:.2f}',
        "p99tpot": f'{d["p99_tpot_ms"]:.2f}',
        "itl": f'{d["mean_itl_ms"]:.2f}',
        "p99itl": f'{d["p99_itl_ms"]:.2f}',
    }

rows = []
if os.path.exists(report_path):
    for line in open(report_path, errors="ignore"):
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) != 9 or not parts[0].isdigit():
            continue
        con, inp, out = map(int, parts[:3])
        metric = {
            "success": parts[3],
            "out": parts[4],
            "total": parts[5],
            "ttft": parts[6].split("/")[0].strip(),
            "p99ttft": parts[6].split("/")[1].strip() if "/" in parts[6] else "-",
            "tpot": parts[7].split("/")[0].strip(),
            "p99tpot": parts[7].split("/")[1].strip() if "/" in parts[7] else "-",
            "itl": parts[8].split("/")[0].strip(),
            "p99itl": parts[8].split("/")[1].strip() if "/" in parts[8] else "-",
        }
        rows.append((con, inp, out, metric, "report"))

for path in glob.glob(os.path.join(result_dir, "*_8b_con=*.log")):
    base = os.path.basename(path)
    m = re.search(r"con=(\d+)_in=(\d+)_out=(\d+)", base)
    if not m:
        continue
    metric = parse_log(path)
    if metric["success"] == "-":
        continue
    rows.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), metric, "official-log"))

for path in glob.glob(os.path.join(param_dir, "*.json")):
    d = json.load(open(path))
    con = int(d["max_concurrency"])
    n = int(d["num_prompts"])
    input_len = round(d["total_input_tokens"] / n)
    output_len = round(d["total_output_tokens"] / n)
    if con in (16, 64) and input_len in (255, 256) and output_len in (255, 256, 1023, 1024):
        rows.append((con, 256, 256 if output_len < 512 else 1024, parse_json(path), "param-json"))

best = {}
for con, inp, out, metric, src in rows:
    key = (con, inp, out)
    score = float(metric["out"]) if metric["out"] != "-" else -1.0
    old = best.get(key)
    if old is None or score > old[0]:
        best[key] = (score, metric, src)

selected = [
    (1, 32, 256),
    (4, 4096, 1024),
    (16, 32, 256),
    (16, 256, 256),
    (64, 32, 256),
    (64, 256, 256),
    (64, 256, 4096),
]

print("con input output success out_tok/s total_tok/s mean/p99_TTFT mean/p99_TPOT mean/p99_ITL source")
for key in selected:
    item = best.get(key)
    if item is None:
        print(*key, "MISSING")
        continue
    _, m, src = item
    success = m["success"]
    if "/" not in success:
        success = f"{success}/{success}"
    print(
        f"{key[0]:>3} {key[1]:>5} {key[2]:>6} {success:>7} "
        f"{m['out']:>9} {m['total']:>11} "
        f"{m['ttft']}/{m['p99ttft']} "
        f"{m['tpot']}/{m['p99tpot']} "
        f"{m['itl']}/{m['p99itl']} {src}"
    )
PY
