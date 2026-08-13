ok/flagscale‑pytorch‑model‑recovery‑test
PyTorch 模型恢复
```
概述
该任务要求智能体从已保存的状态字典逆向还原 PyTorch Transformer 模型架构，加载预训练权重，之后仅对指定层进行微调，提升模型在给定数据集上的性能。
智能体必须完成以下工作：
分析/app/weights.pt，还原出精确的模型架构（编码器‑解码器结构的 Transformer）
定义与状态字典结构完全匹配的RecoveredModel类
加载原始权重，在/app/dataset.pt上计算基准均方误差（MSE）损失
仅对output_layer权重执行微调以降低均方误差损失（其余所有层必须保持冻结状态）
将优化后的模型以 TorchScript 格式保存至/app/model.pt
考察技能
PyTorch 模型架构还原：依据状态字典的键与张量维度推断模型结构
迁移学习：冻结部分网络层，同时对其他层做微调
模型优化：使用梯度下降降低数据集上的损失值
TorchScript 转换：将模型保存为可投入生产环境的格式
深度学习调试：理解 Transformer 架构、位置编码以及注意力机制
环境
基础镜像：python:3.13‑slim‑bookworm
预装工具：torch==2.7.1、nano
工作目录：/app
资源配置：1 核 CPU、2GB 内存、10GB 存储空间
提供文件：
/app/weights.pt — 预训练模型状态字典
/app/dataset.pt — 用于训练的输入‑输出序列配对数据
验证项
测试套件（test.sh调用test_outputs.py）会校验以下内容：
文件完整性：原始weights.pt文件未被改动
模型是否存在：/app/model.pt存在，且为合法的 TorchScript 模型
权重兼容性：保存后的模型可以无报错加载原始权重
选择性训练校验：除output_layer以外，其余全部层权重与原始权重完全一致
性能提升：微调后模型的均方误差损失低于基准损失
必须通过全部 5 项测试才算任务成功，以此证明代码逻辑正确且模型性能得到提升。
```


 
