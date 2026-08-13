# IMDB BERT 端到端微调任务（Hugging Face Parquet）

你正在运行一个容器里面，可见资源包含：
- Python 3
- pip 与 venv
- shell 工具
- 网络访问能力
- 1 块可见 NVIDIA GPU
- 本地挂载的 IMDB 数据集目录：/datasets/imdb
该数据集为 Hugging Face 本地格式， 以parquet 文件格式呈现：
```text
train-00000-of-00001.parquet
test-00000-of-00001.parquet
unsupervised-00000-of-00001.parquet
```
你必须先自动识别数据格式，并从本地路径 /datasets/imdb 读取数据。
不得重新从互联网下载 IMDB 数据集。

你的任务是端到端完成一个英文情感二分类训练任务（IMDB）：
- 标签：positive / negative
- 模型：BERT（二分类）
- 使用 GPU 训练
- 划分：使用本地 train/test
- 输出完整训练结果与评测结果

请完成以下事项：
1. 检查系统与 GPU 环境；
2. 创建独立 Python 虚拟环境；
3. 创建独立环境 /opt/imdb-bert-env，并安装至少包括：CUDA PyTorch、Transformers、Datasets、PyArrow、Tokenizers、Safetensors、scikit-learn 等训练与评测依赖；
4. 自动识别并读取 /datasets/imdb 中的数据；
5. 完成 BERT 模型二分类训练
实验设置：
训练/验证 = 20000/5000
测试 = 25000
```
seed = 42
max_length = 256
epochs = 3
batch size = 16
learning rate = 2e-5
weight decay = 0.01
warmup ratio = 0.1
max_grad_norm = 1.0
optimizer = AdamW；
```
注意一轮epoch是要完整的跑完整个训练集中所有的数据，之后在计算loss，再接着跑知道设定的epochs

6. 在测试集上推理；
7. 在 `/app/submission` 生成：

```text
setup_env.sh
train.py
evaluate.py
run_manifest.json
metrics.json
runtime.json
training.log
predictions.jsonl
model/config.json
model/tokenizer_config.json
model/model.safetensors
```

模型和 tokenizer 必须能用 `local_files_only=True` 加载。Verifier 会独立读取 Parquet，
对全部 25,000 条 test 记录重新推理。
`run_manifest.json` 记录实验中参数等设置，其中 `dataset_path` 必须为`/datasets/imdb`，并记录 `train_file`、`test_file`、数据规模、随机种子、超参数和依赖版本。

`metrics.json` 至少记录 accuracy、precision、recall、f1、roc_auc。
`runtime.json` 至少记录 setup_sec、train_sec、eval_sec、peak_gpu_memory_mb、device。
`predictions.jsonl` 必须有 25,000 行，ID 使用 `test:<row_index>`。

安装脚本必须在以下路径提供可执行的 Python 解释器：
/opt/imdb-bert-env/bin/python
校验程序将调用该确切路径

此外，必须实际完成微调和测试。
