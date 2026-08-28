# P2 Cross-encoder 真实模型验证记录

验证日期：2026-08-28  
验证环境：macOS x86_64，Python 3.13，CPUExecutionProvider  
模型：`Xenova/ms-marco-MiniLM-L-6-v2` 的 ONNX 导出版本  

## 验证配置

- 模型目录通过 `RAGLOOKER_RERANKER_MODEL_DIR` 注入。
- 运行时文件：`model.onnx`、`tokenizer.json`。
- 重排范围：混合召回后的最多 50 个候选。
- 项目真实数据：FAISS 30,693 个向量、BM25 39,175 个游戏。
- Ollama 在本机未运行，因此本次链路验证使用 BM25 召回和确定性生成回退；不影响 ONNX 重排推理验证。

## 结果

| 查询 | 候选数 | 中位推理耗时 | 最慢一次 | 结果 |
|---|---:|---:|---:|---|
| relaxing single player farming game | 50 | 0.676 s | 0.686 s | 通过 |

验收结果：

- 模型成功加载，状态为 `available`。
- 50 个候选均获得一个分数。
- 分数经过 sigmoid 归一化，全部位于 `[0, 1]`。
- 重排前后候选集合完全一致，满足 Recall@50 不因重排下降的要求。
- 输入候选顺序未被推理函数修改。
- 50 对推理最慢 0.686 秒，低于 15 秒预算。
- 真实整条搜索链路已验证 `reranker_applied=true`，并正常返回推荐结果。

## 语义冒烟测试

使用 3 个构造候选集进行真实模型推理：

- `a relaxing single player farming game` → `Cozy Farm` 第一名
- `competitive multiplayer combat game` → `Tactical Strike` 第一名
- `thoughtful puzzle adventure` → `Puzzle Adventure` 第一名

这表明模型能够利用查询与候选描述、标签和模式之间的语义对应关系完成基本重排。

## 结论与限制

P2 的模型加载、真实推理、延迟预算、候选集合不变和故障回退验证均通过，可以进入后续离线质量评测。当前未将模型权重提交到仓库；部署环境需要单独准备模型目录并设置环境变量。此次验证尚不能证明 nDCG、MRR 等离线质量指标提升，需在 P3 评测集上继续验证。
