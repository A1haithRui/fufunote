# PR 提交清单

## 建议提交内容

代码改动：

- `python/infinilm/llm/scheduler.py`

文档材料：

- `doc_extern/T2-1-2_8B_服务能力优化报告.md`
- `doc_extern/HONOR_CODE.md`
- `doc_extern/REFERENCE.md`
- `doc_extern/PR提交清单.md`

可作为补充材料但不一定放入 PR 的实验记录：

- `doc_extern/当前baseline记录与卡点分析.md`
- `doc_extern/推理赛道2优化思路.md`
- `doc_extern/环境配置相关记录.txt`
- `/root/autodl-tmp/bench-results-official-8b/`
- `/root/autodl-tmp/bench-results-param-scan/`

## 不建议提交内容

- `third_party/json`
- `third_party/spdlog`
- `python/infinilm/llm/scheduler.py.bak_20260711_longctx`
- `python/infinilm/llm/scheduler.py.bak_20260711_phasecap`
- `doc_extern/codex_write_test.txt`
- 大量 benchmark 原始 JSON/log，除非比赛明确要求随 PR 附件提交

## PR 摘要建议

本 PR 针对单卡 RTX 4090 24GB 上的 Llama-3.1-8B-Instruct 服务能力进行调度优化。原始实现会在长输入或高并发 prefill/decode 交叠时触发 `cudaMalloc failed`，导致 RankWorker 停止和服务 unhealthy。改动通过分离 prefill/decode batch 上限、增加长输入保护和 decode-first 保护，在保持短输入高并发吞吐的同时，使官方 8B 矩阵 30 个测试点全部稳定完成。

## 关键结果

- `con=64, input=32, output=256`：1052.16 output tok/s。
- `con=64, input=256, output=256`：725.92 output tok/s。
- `con=4, input=4096, output=1024`：从原始 unhealthy 变为 40/40 成功。
- `con=16, input=256, output=256`：从原始 failure 风险点变为稳定完成，分相位限批后输出吞吐从 407.30 tok/s 提升到 726.03 tok/s。

## 提交前检查

- 确认远端服务 `/health` 为 healthy。
- 确认 `doc_extern/T2-1-2_8B_服务能力优化报告.md` 中官方 8B 30 点矩阵完整。
- 确认 `git diff -- python/infinilm/llm/scheduler.py` 只包含调度器相关改动。
- 提交时只 stage 建议提交内容，避免带入 third_party 脏改动和备份文件。
