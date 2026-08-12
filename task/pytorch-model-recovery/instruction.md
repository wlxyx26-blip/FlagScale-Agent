你正在运行在一个容器中，可见资源：
- Python 3
- pip 与 venv
- shell 工具
- 网络访问能力

现有一份 PyTorch 状态字典（路径：/app/weights.pt），存储着 PyTorch 模型权重，另有数据集文件（路径：/app/dataset.pt），包含多组输入 - 输出数据对。
先给你的任务要求如下：
任务：
◦ 准备环境，需要torch==2.7.1 
◦ 利用 /app/weights.pt 内的信息还原原始模型架构。你需要定义 RecoveredModel 类，该类结构必须与这份状态字典体现的模型结构完全一致。
◦ 将 /app/weights.pt 中的原始权重加载至模型，并在 /app/dataset.pt 提供的数据集上计算模型的均方误差（MSE）损失。
◦ 仅调整 "output_layer" 层的权重，使均方误差损失低于使用 /app/weights.pt 原始权重得到的损失。模型其余所有层必须保持固定（冻结参数）。调优完成后，在同一数据集上重新计算均方误差损失。
◦ 将更新权重后的模型以 TorchScript 格式保存至文件 /app/model.pt。

成功判定标准：
・/app/model.pt 中的 TorchScript 模型能够无报错加载 /app/weights.pt 里的原始权重。
・/app/model.pt 与 /app/weights.pt 二者状态字典之间，仅允许 output_layer 层权重存在差异。
・使用调优后的 output_layer 计算得到的均方误差损失，必须低于 /app/weights.pt 未修改权重对应的原始损失。
・禁止修改 /app/weights.pt 文件。

除生成 /app/model.pt 外，还需要保存以下文件：

- /app/recovered_model.py：完整的模型结构还原与训练代码；
- /app/recovery_metrics.json：记录原始MSE、调优后MSE、训练轮数和运行时间；
- /app/model.pt：最终TorchScript模型。

recovery_metrics.json格式示例：

{
  "baseline_mse": 0.123,
  "tuned_mse": 0.087,
  "epochs": 10,
  "train_sec": 15.6
}