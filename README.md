# 用公开数据学习 AI 与城市规划

这个项目用西雅图的公开街道数据回答一个小而完整的问题：

**如果只能调查一部分街道，机器学习能否帮助找到更多有公开障碍记录的街道？如果同时要求照顾不同社区，会有什么变化？**

项目包含下载好的真实数据、可运行代码、已运行 Notebook、分析结果和中文学习材料。无需申请地图 API、采集个人轨迹或购买 GPU。

## 从这里开始

1. 阅读 [学习路线](LEARNING_GUIDE.md)。先学第一课即可，不要求先读完规划理论。
2. 打开 [已运行 Notebook](START_HERE.ipynb)，看每段问题、代码和输出。
3. 阅读 [本次实验报告](REPORT.md)，区分实际发现与仍不知道的事情。
4. 修改一个参数，重跑并写下解释。练习说明和答案提示在学习路线中。

当前版本是一个完整的公开数据学习基线。它研究**调查名额如何分配**，暂不估计老年人步行时间、医疗服务覆盖、工程造价或道路改造的因果效果。

## 安装与运行

推荐 Python 3.11，在本文件所在目录运行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis.py
```

Windows 的虚拟环境激活命令为 `.venv\Scripts\activate`。

数据快照已随项目提供，默认分析不联网。要重新下载平台当前数据：

```bash
python download_data.py --refresh
python analysis.py
```

平台全城导出可能需数分钟，下载脚本设置了超时。刷新会改变数据版本，因此结果可能与随附报告不同；刷新后应重新解释结果，不能继续引用旧报告的数字。

Notebook 可以用 Jupyter、VS Code 或支持 `.ipynb` 的编辑器打开。也可以在命令行执行：

```bash
python make_notebook.py
```

这个命令重建并执行学习 Notebook；默认读取已生成的实验结果，不重新训练。完整训练由 `analysis.py` 执行。

## 文件怎么对应

| 路径 | 内容 |
|---|---|
| `START_HERE.ipynb` | 按学习顺序组织、保留输出的 Notebook |
| `LEARNING_GUIDE.md` | 中文解释、公式、练习与判断标准 |
| `REPORT.md` | 真实运行结果及证据边界 |
| `DATA_CARD.md` | 来源、许可证、目标定义和抽样限制 |
| `analysis.py` | 特征提取、训练、验证、调查策略比较、绘图 |
| `download_data.py` | 公开数据下载与 SHA-256 记录 |
| `data/raw/*.geojson.gz` | 压缩的原始公开数据快照 |
| `data/raw/manifest.json` | 下载 URL、时间、条数、哈希 |
| `data/processed/streets.csv` | 供学习和分析的街道表 |
| `results/` | 指标、逐路段预测、方案选择和图表 |
| `checks.py` | 数据隔离、数值与产物的一致性检查 |

## 数据来源

Project Sidewalk 的道路、聚合标签与设施 API：

- https://sidewalk-sea.cs.washington.edu/v3/api-docs/streets
- https://sidewalk-sea.cs.washington.edu/v3/api-docs/labelClusters
- https://sidewalk-sea.cs.washington.edu/v3/api-docs/places

障碍标注采用 CC0；OSM 衍生道路与设施遵循 ODbL，© OpenStreetMap contributors。原始街景图像不包含在项目中。详见 [数据说明](DATA_CARD.md)。

本项目中的代码和文稿由 AI 辅助生成，统计结果由随附脚本实际计算。学习者应逐项理解和复核；用于申请时，应如实说明自己的实施、验证与扩展工作。
