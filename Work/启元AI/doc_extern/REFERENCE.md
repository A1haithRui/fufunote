# REFERENCE

本次提交未复制外部开源项目的核心实现代码。优化工作基于比赛提供的 InfiniLM 仓库现有调度器实现，在 `python/infinilm/llm/scheduler.py` 内进行小范围修改，并通过 vLLM benchmark 工具进行服务能力测试。

## 1. 比赛提供的 InfiniLM 仓库

- 资源名称：InfiniLM
- 资源链接：比赛官方仓库 / 本次 PR 所在仓库
- 参考内容：`python/infinilm/llm/scheduler.py` 中已有 waiting/running queue、Paged Attention KV cache admission、prefill/decode scheduling 流程。
- 修改与优化：在原有调度流程上增加可配置的 prefill/decode 分相位 batch cap、长输入 batch 保护、decode-first 保护；未引入外部项目代码。
- 协议：以比赛仓库原协议为准。

## 2. vLLM benchmark serve

- 资源名称：vLLM benchmark serve
- 资源链接：https://github.com/vllm-project/vllm
- 参考内容：使用 `vllm bench serve` 作为压测工具，采集 TTFT、TPOT、ITL、吞吐等指标。
- 修改与优化：未修改 vLLM 源码，仅使用其命令行工具进行性能评估。
- 协议：Apache-2.0。

## 3. OpenAI Codex

- 资源名称：OpenAI Codex
- 资源链接：https://openai.com/codex
- 参考内容：用于协助整理实验记录、分析 benchmark 结果、撰写报告，并辅助生成小范围代码改动方案。
- 修改与优化：最终代码与实验决策均由参赛团队确认；未从 Codex 输出中复制第三方开源项目代码。
- 协议：不涉及开源代码复用。
