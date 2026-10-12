# Homework 1：NYT 新闻文本分类

南开大学自然语言处理课程作业一，比较词袋、平均词向量与 BERT 微调三类文本分类方法。

**实验报告：** [中文版](report/实验报告_中文版.pdf) · [英文版](report/实验报告.pdf)

## 主实验

使用 [WeSHClass NYT 数据](https://github.com/yumeng5/WeSHClass/tree/master/nyt)。去重并移除冲突标注后，共 **13,007 篇新闻、29 个类别**；以 `seed=42` 按 80% / 10% / 10% 划分训练、验证和测试集，所有方法共享划分。

主实验测试结果如下，指标单位为 %：

| 任务 | 方法 | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| Task 1 | Binary BoW + LR | 92.47 | 80.47 |
| Task 1 | 词频 + LR | 93.63 | 83.61 |
| Task 1 | TF-IDF + LR | 94.01 | 82.85 |
| Task 2 | [GloVe 6B](https://nlp.stanford.edu/projects/glove/) + LR | 91.71 | 79.38 |
| Task 2 | [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) Word2Vec + LR | 90.02 | 77.11 |
| Task 2 | NYT Word2Vec + LR | 92.63 | 81.85 |
| Task 3 | [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) | 89.63 | 62.97 |

LR 为逻辑回归。Task 2 使用 100 维词向量的平均池化；Task 3 的最大输入长度为 64，训练 3 轮。词表、IDF、NYT Word2Vec 和特征标准化仅在训练集拟合，参数或检查点按验证集 Macro-F1 选择。完整设置与分析见报告。

## 复现

推荐 Python 3.10，BERT 训练建议使用 GPU。从仓库根目录执行：

```bash
cd homework1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 下载 NYT、AG News、GloVe 和 BERT 资源
python scripts/prepare_assets.py

# 运行 Task 1–3，并核验预测与指标
PYTHONHASHSEED=42 python run_experiments.py --bert-model data/bert-base-uncased
python scripts/verify_results.py
```

后续命令均在 `homework1/` 目录执行。常用选项：

- `--tasks bow`、`--tasks embeddings`、`--tasks bert`：单独运行对应任务。
- `--device cpu`：使用 CPU；多 GPU 环境可通过 `CUDA_VISIBLE_DEVICES` 选择设备。
- `--nyt <文件路径>`：使用自定义数据，需包含 `text`、`label` 两列。

下载资源保存在 `data/`，主实验结果写入 `outputs/main/`，汇总指标见 `results.csv`。数据、模型权重和运行产物不纳入仓库。

## 扩展实验（可选）

完成主实验后，依次运行：

```bash
# 固定划分：输入长度、类别加权、词向量池化和训练种子
PYTHONHASHSEED=42 python -m scripts.run_explorations
python scripts/verify_results.py --explorations outputs/explorations

# 重复划分：损失对照、训练预算和文本位置
PYTHONHASHSEED=42 python -m scripts.run_scaling
python scripts/verify_scaling.py
```

结果分别保存在 `outputs/explorations/` 和 `outputs/scaling/`；实验设计与结果解读见报告。
