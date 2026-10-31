# 复现与验证

## 1. 复核公开聚合结果（CPU）

在仓库根目录运行，Python 3.10+，无第三方依赖：

```bash
python scripts/summarize_results.py
python -m unittest discover -s tests -v
```

这些操作读取已经发布的聚合值并检查分母、计数、范围和版本关系，不重新运行评测。原始聚合文件的来源与 SHA 记录在 `results/provenance.json`。

## 2. 重画结果图（CPU）

```bash
python -m pip install -r requirements-plot.txt
python scripts/plot_results.py
```

生成 `assets/results.png` 与 `assets/results.svg`。两轮 40 任务实验分开呈现；没有统计置信区间时不绘制伪误差棒。

## 3. 理解决策规则（CPU）

```bash
python examples/route_demo.py
```

演示使用合成分数，展示锚点、冻结阈值和独立随机流。它不是来自真实 hidden 的新成功率或延迟测试。

`routing/risk_mlp.py` 依赖 PyTorch。请根据本机平台采用 [PyTorch 官方安装方式](https://pytorch.org/get-started/locally/)。这里只提供真实 MLP 结构和输入 mask；没有附带经过训练的模型检查点。基础 CI 不安装 PyTorch，因此不会验证真实 MLP 前向或重训结果。

## 4. 完整模型实验所需的额外材料

完整闭环重跑还需要官方 FastWAM 模型和代码、LIBERO 环境及资产、对应版本的浅层 adapter 和 router 检查点、采集特征与监督标签、冻结校准配置、初态列表，以及兼容 GPU 环境。本发布包未提供完整 GPU runner 和这些资产，不能声称开箱即用地复现论文式完整实验。

Task4 原训练为每 suite 16 个采集 episode，按 episode 分为 10 训练 / 2 验证 / 2 校准 / 2 诊断；不得改为按单行随机切分。使用训练集归一化，200 次 AdamW 更新，学习率 1e-3，weight decay 1e-4，梯度裁剪 1.0，训练 seed61042。该训练描述不适用于历史 40 任务 MLP 的所有设置。

如补充完整重跑入口，应使用新的发布版本，明确资产版本和缺失项；不得以合成数据运行结果冒充归档指标。
