### 使用教程

### 仓库说明
本仓库提供基于 Harbor 的 FlagScale-Agent 自定义评测任务，
用于评估 Agent 在机器学习训练、模型恢复等真实终端任务中的：

- 最终任务正确性；
- 任务产物质量：工具使用和执行过程质量和异常识别与恢复能力。

FlagScale-Agent 执行过程中生成原生 JSONL trace，Harbor Adapter 会在任务结束后将其转换为 ATIF-v1.7，供 LLM-as-Judge 进行过程质量评测。

### 前置条件
- **系统：** Linux
- **硬件：** GPU/CPU

> **运行说明：** 建议在本地运行时，将 `FlagScale-Agent` 源码（除去`task`外的代码文件）与 本仓库中`task` 任务放在不同文件夹中。例如：

```text
Test/
├── FlagScale-Agent/
├── task/
│   ├── data/
│   ├── harbor_adapter/
│   ├── imdb-bert-train/
│   └── ...
```
> **代码说明：** 相较于官方FlagScale-Agent代码仓库，本仓库中针对FlagScale-Agent源码修改了flagscale_agent/react/agent.py、FlagScale-Agent/flagscale_agent/react/providers/anthropic_provider.py文件，并添加FlagScale-Agent/flagscale_agent/trace_logger.py文件，获取FlagScale-Agent的原生轨迹文件，以便后续生成harbor框架接受的用于LLM-As-Judeg判断的ATIF轨迹文件

### 1. 安装 Harbor

首先安装 `pipx`：

```bash
apt install -y pipx
```

使用 `pipx` 安装 Harbor：

```bash
pipx install harbor --pip-args="-i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 300"
```

> 建议使用清华 PyPI 镜像，以提高下载速度。

将 `pipx` 安装路径添加到 `PATH`：

```bash
pipx ensurepath
source ~/.bashrc
```

检查 Harbor 是否安装成功：

```bash
harbor --version
```

### 2. 自定义接口

Harbor 支持集成用户自定义的 Agent 程序，当前仓库已完成 `FlagScale-Agent` 与 Harbor 的自定义接口适配，`task/harbor_adapter/flagscale_agent_adapter`

### 3.测评任务
运行时，任务文件需要修改成本地路径以及相其他关内容。

### imdb-bert-train

修改`imdb-bert-train/environment/docker-compose.yaml`文件
```bash
volumes:
      - {FlagScale-Agent Path}:/mnt/flagscale-agent-host:ro  #添加本地agent源码所在路径
      - {Imdb data Path}:/datasets/imdb:ro  #添加本地数据集路径
      - {HF cache Path}:/root/.cache/huggingface:ro  #添加本地HF cache 路径 例如：/root/.cache/huggingface
 environment:
      HF_HOME: {HF cache Path} #添加本地HF cache 路径
      TRANSFORMERS_CACHE: {HF cache Path}/hub  #添加存储的模型文件缓存位置，降低重复下载
      TOKENIZERS_PARALLELISM: "false"
      HTTP_PROXY: http://用户名:密码@代理地址:80   #添加本地VPN
      HTTPS_PROXY: http://用户名:密码@代理地址:80
      NO_PROXY: localhost,{本机 ip},{api 访问地址}   #添加容器访问哪些地址要跳过VPN，直接链接，例如：本机ip，模型推理请求的服务访问地址，以免报错

device_ids:
  - "${IMDB_GPU_DEVICE_ID}"   #当前任务声明task.toml是需要1张GPU,这里是声明使用gpu标号，不需要可删除

```
修改`imdb-bert-train/task.toml`文件，添加判定任务质量的LLM
```bash
[verifier.env]
#若使用opai
OPENAI_API_KEY = "your_kpi"
OPENAI_API_BASE = "kpi_base_url"

#若使用anthropic
ANTHROPIC_API_KEY = "your_kpi"
ANTHROPIC_BASE_URL = "kpi_base_url"

```
对应`imdb-bert-train/tests/quality/quality.toml`也要修改，替换成要使用的模型，例如：openai/gpt-5.6-sol
```bash
[judge]
judge = "model name"
```

### pytorch-model-recovery

修改`pytorch-model-recovery/environment/docker-compose.yaml`文件
```bash
volumes:
      - {FlagScale-Agent Path}:/mnt/flagscale-agent-host:ro  #添加agent源码所在路径

environment:
      HF_HOME: {HF cache Path} #添加HF cache 路径
      TRANSFORMERS_CACHE: {HF cache Path}/hub  #添加存储的模型文件缓存位置，降低重复下载
      TOKENIZERS_PARALLELISM: "false"
      HTTP_PROXY: http://用户名:密码@代理地址:80   #添加VPN
      HTTPS_PROXY: http://用户名:密码@代理地址:80
      NO_PROXY: localhost,{本机 ip},{api 访问地址}   #添加容器访问哪些地址要跳过VPN，直接链接，例如：本机ip，模型推理请求的服务访问地址
```

### 4.准备镜像
### local/flagscale-agent-base:cu128-py312
```bash
在命令行输入：
docker pull harbor.baai.ac.cn/flagscale/flagscale-train:dev-cu128-py3.12-20260319182856

IMAGE="harbor.baai.ac.cn/flagscale/flagscale-train:dev-cu128-py3.12-20260319182856"
LOCAL_IMAGE="local/flagscale-agent-base:cu128-py312"

docker tag "$IMAGE" "$LOCAL_IMAGE"
```
### local/flagscale-agent-base-rewardkit:cu128-py312
```bash
单独创建Dockerfile.rewardkit-base文件,编写下面内容:
FROM local/flagscale-agent-base:cu128-py312
RUN python3 -m venv /opt/rewardkit-env && \
    /opt/rewardkit-env/bin/python -m pip install --no-cache-dir \
        "harbor-rewardkit==0.1.7"
ENV PATH="/opt/rewardkit-env/bin:${PATH}"

在命令行输入:
docker build \
  -f Dockerfile.rewardkit-base \
  -t local/flagscale-agent-base-rewardkit:cu128-py312 .
```
> 成功创建可以LLM-as-judge环境的公共镜像，如imdb-bert-train

### 4.运行
使用命令：
```bash
cd task

PYTHONPATH="$PWD" harbor run   \
-p 测评任务路径
--override-gpus 0 \
--agent-setup-timeout-multiplier 10       \
--agent harbor_adapter.flagscale_agent_adapter:FlagScaleInstalledAgent  \
--ae FLAGSCALE_PROVIDER="anthropic"  \
--ae ANTHROPIC_BASE_URL="your_kpi_base_url"   \  
--ae ANTHROPIC_API_KEY="your_api_key"  \
--ae ANTHROPIC_MODEL="model name"  \
--ae FLAGSCALE_TRACE_ENABLED="1"  \
--ae FLAGSCALE_TRACE_PATH="/logs/agent/flagscale-events.jsonl"  \

#当前命令运行FlagScale-Agent使用ANTHROPIC，可以替换成opai
```

### 5.评测任务流程
```bash
   命令
    ↓
启用容器、agent
    ↓
 instruction.md
    ↓
 FlagScale-Agent自行完成任务
    ↓
 生成/app/submission 等文件，可设置保存到宿主机
    ↓
 tests/test.sh运行
    ↓
 test_outputs.py、quality.toml等文件，验证结果or质量
    ↓
 生成/logs/verifier/reward.json等文件
 ```
