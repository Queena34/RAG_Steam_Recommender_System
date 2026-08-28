# P3 离线评测与可观测性报告

验证日期：2026-08-28  
数据集：`eval/queries_knownitem.jsonl` 前 20 条 Known-item 查询  
模型：`Xenova/ms-marco-MiniLM-L-6-v2` ONNX 导出版本  
运行环境：macOS x86_64，CPUExecutionProvider；Ollama 未运行，因此召回使用 BM25

## 配对结果

| 配置 | Recall@50 | nDCG@10 | MRR | P@5 | ILD@5 | 平均查询耗时 |
|---|---:|---:|---:|---:|---:|---:|
| `full_no_rerank` | 0.100 ± 0.069 | 0.047 ± 0.035 | 0.031 ± 0.025 | 0.010 ± 0.010 | 0.916 ± 0.009 | 2.85 s |
| `full_rerank` | 0.100 ± 0.069 | 0.019 ± 0.019 | 0.014 ± 0.011 | 0.010 ± 0.010 | 0.914 ± 0.007 | 5.68 s |

Recall@50 在 20/20 条配对查询中一致，但当前样本中 Cross-encoder 的 nDCG@10 和 MRR 下降，因此暂不能宣称质量提升。平均查询耗时增加约 2.83 秒；单次真实 API 测试记录重排耗时 2.87 秒、总耗时 16.06 秒。

## 已接入的可观测性

`/api/search` 的 `meta` 现在包含 `candidate_count`、`retrieval_latency_ms`、`reranker_latency_ms`、`reranker_model`、`reranker_status`、`reranker_applied` 和 `total_latency_ms`。评测 harness 提供 `full_no_rerank` 与 `full_rerank` 配置，并在无 Ollama 时自动跳过不可用的 vector retriever。

可通过 `RAGLOOKER_RERANKER_ENABLED=0` 关闭重排，执行线上或离线对照。

## 模型对比与后续实验

同样的 ONNX 接口已成功加载 `bge-reranker-base`，但在当前 Intel CPU 上一次 50 候选批量推理超过 60 秒，未达到 15 秒预算，因此没有把未完成的 BGE 结果伪装成质量对比。MiniLM 是当前唯一完成延迟与样本评测的候选。

补充验证：Ollama 恢复后，完整 FAISS + BM25 已确认启用。50 条 Known-item 的无重排基线完成，Recall@50 为 `0.120 ± 0.046`、nDCG@10 为 `0.024 ± 0.014`。同样的 50 条重排组在本机因 Ollama embedding 与 CPU Cross-encoder 组合耗时过长，未完成，因此没有报告不完整的重排指标。500 条全量任务也确认需要改造 BM25/SQLite 缓存或迁移到更强的离线计算环境。

评测脚本新增 `eval/analyze_p3.py`，可按 `category` 汇总配对差异；`eval/run_eval.py` 已记录每条查询的 category 和 latency，后续可直接执行完整 Track A/B。权重扫描仍应在完整候选结果上进行，不能基于当前 20 条样本直接重新标定。

## 决策

P3 的离线评测和可观测性基础设施已完成，但 Cross-encoder 质量门禁暂不通过。模型继续保持 opt-in。下一步应扩充分层 Track A/Track B 评测，检查候选文本模板与模型适配，并重新评估排序权重。
