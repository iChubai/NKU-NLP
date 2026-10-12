# Homework 1：NYT 新闻文本分类

南开大学自然语言处理课程作业一。仓库总览见 [`../README.md`](../README.md)。

学生：侯嘉栋（Jiadong Hou），学号：2311671。

## 实验报告

- [英文实验报告](report/实验报告.pdf)
- [中文实验报告](report/实验报告_中文版.pdf)

报告按 Task 1、Task 2、Task 3 组织，包含各项任务的原理、设置、结果与分析，以及扩展实验和图表。

## 课程任务

所有方法共享同一训练、验证和测试划分，使用 Accuracy 与 Macro-F1 评价。

| 任务 | 实验方法 | 实现 |
| --- | --- | --- |
| Task 1 | Binary BoW、词频、TF-IDF + Logistic Regression | `textcls/classical.py` |
| Task 2 | GloVe 6B、AG News Word2Vec、NYT Word2Vec 的 100 维普通平均池化 + Logistic Regression | `textcls/classical.py` |
| Task 3 | BERT-base-uncased，最大输入长度 64，完整训练 3 轮 | `textcls/neural.py` |

扩展实验比较输入长度、类别加权、文本位置、词向量池化、训练预算，以及多划分和多训练种子的波动。

## 数据来源与预处理

作业要求没有附带 NYT 数据文件或下载地址。本项目使用 [WeSHClass 的公开 NYT 数据](https://github.com/yumeng5/WeSHClass/tree/master/nyt)，将 `dataset.txt` 与 `labels.txt` 按行对应，保留原始 29 个主题标签。其他资源为：

- [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) 的 120,000 条训练文本，仅用于训练 Word2Vec，标签不参与训练。
- [Stanford GloVe 6B](https://nlp.stanford.edu/projects/glove/) 的 100 维词向量，使用 Gensim 的 `glove-wiki-gigaword-100` 镜像下载。这是 Wikipedia 2014 + Gigaword 5、6B tokens、400,000 个词的同一组预训练向量；解压时移除 word2vec 格式的首行。
- [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)。

原始 NYT 有 13,081 条记录。划分前移除 2 组同一文本具有不同标签的记录（共 4 条），对其余文本去重，移除 70 条重复记录，得到 13,007 篇新闻。这样同一文本不会跨越训练、验证和测试集合。

设置 `seed=42`，随机打乱清洗后的行号，按 80% / 10% / 10% 划分，实际为 **10,405 / 1,300 / 1,302** 篇。所有方法使用同一个 `splits.json`。其中 `row_id` 是预处理后语料的行号。

若老师提供指定版本，输入文件应为 UTF-8 CSV，包含 `text,label` 两列，例如：

```csv
text,label
"The team won the championship.",sports
"Scientists discovered a new planet.",science
```

也支持具有相同列的 JSONL、Parquet。使用 `--nyt` 指定文件，并选择新的 `--output` 目录，重新运行全部实验。

## 安装

推荐 Python 3.10。以下安装、下载、实验与核验命令均在 `homework1/` 目录执行。从仓库根目录开始安装：

```bash
cd homework1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

本次实验使用 PyTorch 2.9.0 + CUDA 12.6、Transformers 4.57.3、scikit-learn 1.7.2、gensim 4.4.0。运行实验会将准确环境版本写入 `outputs/main/environment.json`。需要 CUDA 12.6 版 PyTorch 时，可使用：

```bash
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu126
```

## 下载数据和预训练资源

```bash
python scripts/prepare_assets.py
```

该命令准备 `data/nyt.csv`、`data/ag_news.csv`、`data/glove.6B.100d.txt` 和 `data/bert-base-uncased/`，并记录来源于 `data/sources.json`。已存在的资源直接复用。可选择下载某些资源：

```bash
python scripts/prepare_assets.py --assets ag-news glove bert
```

## 运行 Task 1–3

```bash
PYTHONHASHSEED=42 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python run_experiments.py --bert-model data/bert-base-uncased
```

有多个 GPU 时，使用 `CUDA_VISIBLE_DEVICES` 选择可用设备。例如，选择编号为 0 的 GPU：

```bash
PYTHONHASHSEED=42 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python run_experiments.py --bert-model data/bert-base-uncased
```

CPU 运行可添加 `--device cpu`。主实验入口将 BERT 的 **max_length=64、epochs=3** 固定在代码中。其他默认参数为：batch size 32、AdamW、learning rate 2e-5、weight decay 0.01、10% warmup；使用验证集 Macro-F1 选择最佳 epoch。

传统模型使用训练集词表和 `C ∈ {0.1, 1, 10}` 的验证集选择。词频使用原始计数，TF-IDF 使用平滑 IDF 和 L2 归一化。词向量维度统一为 **100**；文档表示为有效词向量的平均，包含重复词，无向量的词不计入分母。词向量文档特征的 StandardScaler 仅在训练集拟合。

Word2Vec 使用 Skip-gram、window 5、min_count 2、negative 5、10 epochs、workers 1。NYT Word2Vec 只使用 NYT Training Set，AG News Word2Vec 只使用 AG News 的训练文本。

分任务运行时，使用同一输出目录即可累计结果：

```bash
python run_experiments.py --tasks bow
PYTHONHASHSEED=42 python run_experiments.py --tasks embeddings
python run_experiments.py --tasks bert --bert-model data/bert-base-uncased
```

指定课程数据的完整运行示例：

```bash
PYTHONHASHSEED=42 python run_experiments.py \
  --nyt data/nyt_course.csv --output outputs/course \
  --bert-model data/bert-base-uncased
```

## 拓展实验

完成七组主实验后，执行以下命令运行 37 组新增配置，并复用四组主实验作为参照。本节实验使用主实验的同一份划分；训练种子变化不会改变数据划分。

```bash
PYTHONHASHSEED=42 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
python -m scripts.run_explorations
```

仅有一张 GPU 时，移除 `CUDA_VISIBLE_DEVICES=0`。CPU 上的词袋、词向量任务与 GPU 上的 BERT 训练并行执行。拓展实验包括：

- **输入长度与类别加权**：BERT 的输入长度为 64、128、256，分别使用普通交叉熵与类别频次倒数加权的交叉熵；每种设置运行训练种子 42、43、44，形成 18 组配置，均完整训练三轮。加权只使用 NYT 训练集类别数量。
- **相同文本范围的对照**：三种 BoW 分别使用与 BERT 上述三个长度对应的原始文本前缀，共九组。通过 WordPiece 的字符偏移截取文本，每组重新拟合训练集词表和 IDF。
- **词向量池化**：三组 100 维词向量比较平均池化与 TF-IDF 加权平均；IDF 仅在 NYT 训练集拟合，重复词按出现次数计入。
- **Word2Vec 训练波动**：AG News 和 NYT Word2Vec 均比较种子 42、43、44，每个模型分别评价两种池化。GloVe 使用固定预训练向量。

结果位于 `outputs/explorations/results.csv`，含复用参照在内共有 41 组配置。`summary.json` 保存三种子的均值与样本标准差，以及在同一训练种子下将 BERT 长度从 64 改为 256 的配对差值。低频类 F1 是训练样本数不超过 100 的类别的 F1 宏平均，类别名单也保存在该文件中。`input_statistics.json` 记录 WordPiece 长度、截断比例和前缀包含的词数。每个新配置均保存测试预测、逐类指标和模型选择记录。

## 重复划分与大规模控制实验

```bash
PYTHONHASHSEED=42 python -m scripts.run_scaling
```

完整矩阵包含 **204 组配置**，复用已有的 15 组相同配置，新增 **189 组**。连同主实验和固定划分探索，共 **233 组不同实验配置**。默认使用一张 GPU，并用三个 CPU worker 并行运行传统模型；已完成的配置直接复用。

- **划分与训练波动**：划分种子 42、43、44、45、46，均采用相同的随机 80/10/10 协议。每个划分中的 BERT 比较 256/512 长度、两种损失和三个训练种子，共 60 组。
- **传统模型的损失对照**：三种词袋比较全文及匹配的 256/512 词元前缀，分别使用普通与类别加权逻辑回归，每个划分 18 组，共 90 组。NYT Word2Vec 采用 IDF 池化、两种分类损失及三个训练种子，共 30 组。
- **优化预算**：参考划分比较 64/256/512 长度、两种损失、十轮训练和三个训练种子，共 18 组。每组完成十轮，在验证集选择最佳检查点；学习率调度覆盖指定预算。
- **文本位置**：加权 BERT-256 比较前缀、尾部和头尾各半。尾部及头尾组合新增六组，前缀复用对应三组。每组最多保留 254 个文本 WordPiece，头尾组合拼接前后各 127 个。

每个划分重新拟合词表、IDF、标准化、NYT Word2Vec、类别权重与验证选择。稀疏模型的求解器为确定性的，每个设置拟合一次；BERT 与 NYT Word2Vec 使用训练种子 42、43、44。汇总先在划分内平均训练种子，再计算五个划分均值的均值与样本标准差，另保存各划分内的训练标准差。低频类名单由对应训练划分中不超过 100 的类别确定。

使用四张 GPU 并行运行独立 BERT 配置，传统模型使用五个 CPU worker，每个四线程：

```bash
mkdir -p logs
python -m scripts.run_scaling --task prepare
PYTHONHASHSEED=42 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 \
OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python -u -m scripts.run_scaling --task cpu --cpu-workers 5 \
  --cpu-threads 4 > logs/scaling_cpu.log 2>&1 &
for gpu in 0 1 2 3; do
  PYTHONHASHSEED=42 CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 \
  OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  python -u -m scripts.run_scaling --task bert --gpu-workers 4 \
    --gpu-worker-index $gpu > logs/scaling_gpu$gpu.log 2>&1 &
done
wait
python -m scripts.run_scaling --task summarize
python scripts/verify_scaling.py
```

结果与配置位于 `outputs/scaling/results.csv`、`summary.json` 和 `protocol.json`。各划分分别保存 `splits.json`、类别数量、测试预测和模型选择记录。`input_statistics.json` 给出 512 词元的截断和文本保留统计；预处理缓存位于 `outputs/scaling/cache/`。

## 参考结果

公开 NYT 数据、划分种子 42 下的主实验测试结果，单位为 %。BERT 按 Task 3 的长度 64、3 轮设置运行。

| 方法 | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Binary BoW + LR | 92.47 | 80.47 |
| 词频 + LR | 93.63 | 83.61 |
| TF-IDF + LR | 94.01 | 82.85 |
| GloVe 6B + LR | 91.71 | 79.38 |
| AG News Word2Vec + LR | 90.02 | 77.11 |
| NYT Word2Vec + LR | 92.63 | 81.85 |
| BERT-base-uncased | 89.63 | 62.97 |

## 结果核验

完成主实验后，从保存的逐文档预测重新计算指标，核对共享划分、验证集模型选择与 BERT 的完整训练记录：

```bash
python scripts/verify_results.py --results outputs/main
```

完成扩展实验后，核对全部预测、训练集类别权重以及多种子和多划分汇总：

```bash
python scripts/verify_results.py --results outputs/main \
  --explorations outputs/explorations
python scripts/verify_scaling.py --results outputs/scaling
```

## 代码结构

| 路径 | 功能 |
| --- | --- |
| `run_experiments.py` | Task 1–3 的统一入口 |
| `textcls/experiment.py` | 参数、环境记录与主实验结果汇总 |
| `textcls/data.py` | 数据读取、去重、分词与共享划分 |
| `textcls/classical.py` | 词袋、词向量、池化与逻辑回归 |
| `textcls/neural.py` | BERT 微调、类别加权、文本位置与检查点选择 |
| `textcls/metrics.py` | 指标、逐类结果、混淆矩阵与逐文档预测 |
| `scripts/prepare_assets.py` | 下载公开数据与预训练资源 |
| `scripts/run_explorations.py` | 固定划分的输入长度、池化和多种子实验 |
| `scripts/run_scaling.py` | 多划分、四卡分工、训练预算与文本位置实验 |
| `scripts/verify_results.py` | 主实验与固定划分扩展实验核验 |
| `scripts/verify_scaling.py` | 大规模实验与两层波动汇总核验 |

运行时资源和结果分别写入 `data/`、`outputs/`；并行运行日志写入 `logs/`，这些目录均不上传。

本目录仅提交实验代码、依赖与运行说明，以及 `report/` 中的两份最终 PDF。不提交报告生成、绘图、打包脚本、LaTeX 构建材料、原始数据、模型权重、缓存或提交压缩包。
