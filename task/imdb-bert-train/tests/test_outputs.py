#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

TASK_KIND = "bert"
DATASET = Path("/datasets/imdb")
TRAIN_FILE = DATASET / "train-00000-of-00001.parquet"
TEST_FILE = DATASET / "test-00000-of-00001.parquet"
UNSUP_FILE = DATASET / "unsupervised-00000-of-00001.parquet"
SUBMISSION = Path("/app/submission")
BASELINE_FILE = Path("/tests/baselines.json")
VERIFIER_DIR = Path("/logs/verifier")
ARTIFACT_DIR = Path("/logs/artifacts")
AGENT_DIR = Path("/logs/agent") 
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
MAX_LENGTH = 256

SUCCESS_STATUSES = {"success", "cached"}
ISSUE_STATUSES = {"error", "failed", "failure", "blocked", "denied", "capped"}
PROCESS_WEIGHT = 0.5
ROBUSTNESS_WEIGHT = 0.5
DUPLICATE_WINDOW = 3


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def clamp_100(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_parquet(path: Path, split: str) -> list[tuple[str, int, str]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["text", "label"])
    texts = table.column("text").to_pylist()
    labels = table.column("label").to_pylist()
    rows: list[tuple[str, int, str]] = []
    for index, (text, label) in enumerate(zip(texts, labels)):
        label_int = int(label)
        if label_int not in (0, 1):
            raise ValueError(f"invalid label {label_int} in {path.name} row {index}")
        rows.append((f"{split}:{index}", label_int, str(text)))
    return rows


def collect_rows(split: str) -> list[tuple[str, int, str]]:
    if split == "train":
        return _read_parquet(TRAIN_FILE, "train")
    if split == "test":
        return _read_parquet(TEST_FILE, "test")
    raise ValueError(f"unsupported split: {split}")


def check_dataset() -> tuple[float, dict[str, int]]:  #验证数据完整性
    import pyarrow.parquet as pq

    files = {"train": TRAIN_FILE, "test": TEST_FILE, "unsupervised": UNSUP_FILE}
    counts: dict[str, int] = {}
    schema_ok = 1
    for name, path in files.items():
        if not path.exists():
            counts[f"{name}_rows"] = 0
            schema_ok = 0
            continue
        metadata = pq.read_metadata(path)
        counts[f"{name}_rows"] = int(metadata.num_rows)
        if not {"text", "label"}.issubset(set(metadata.schema.names)):
            schema_ok = 0
    counts["schema_ok"] = schema_ok
    valid = (
        counts.get("train_rows") == 25000
        and counts.get("test_rows") == 25000
        and counts.get("unsupervised_rows") == 50000
        and schema_ok == 1
    )
    return float(valid), counts

def compute_metrics(labels: list[int], predictions: list[int], probabilities: list[float]) -> dict[str, float]:  #指标计算
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    count = len(labels)

    accuracy = (tp + tn) / count if count else 0.0 #所有预测中正确的比例
    precision = tp / (tp + fp) if tp + fp else 0.0 #预测为正类中，真正是正类的比例（查准率）
    recall = tp / (tp + fn) if tp + fn else 0.0 #所有正类中，被正确预测出来的比例（查全率）
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0 #precision 和 recall 的调和平均数

    #原理： AUC = 随机取一个正样本和一个负样本，正样本预测概率高于负样本的概率。
    ranked = sorted(zip(probabilities, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in ranked[index:end])
        index = end

    positives = sum(labels)
    negatives = count - positives
    auc = (
        (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
        if positives and negatives else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
        "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
    }


def artifact_checks() -> tuple[float, dict[str, int]]:  #提交文件完整性检查
    required = [
        "setup_env.sh", "train.py", "evaluate.py", "run_manifest.json",
        "metrics.json", "runtime.json", "training.log", "predictions.jsonl",
    ]
    required += ["model/config.json", "model/tokenizer_config.json","model/model.safetensors"]

    checks: dict[str, int] = {}
    for relative in required:
        path = SUBMISSION / relative
        checks[relative] = int(path.exists() and path.stat().st_size > 0)

    score = sum(checks.values()) / max(1, len(checks))  #通过检查数 / 总检查项数
    return score, checks

def protocol_checks(
    manifest: dict[str, Any],
) -> tuple[float, dict[str, int]]:
    """检查训练协议是否符合任务要求，兼容顶层和嵌套超参数结构。"""

    if not isinstance(manifest, dict):
        manifest = {}

    hyperparameters = manifest.get("hyperparameters", {})
    if not isinstance(hyperparameters, dict):
        hyperparameters = {}

    def get_value(*keys: str, default: Any = None) -> Any:
        """优先从manifest顶层读取，其次从hyperparameters读取。"""
        for key in keys:
            if key in manifest:
                return manifest[key]

        for key in keys:
            if key in hyperparameters:
                return hyperparameters[key]

        return default

    def numeric_equal(
        value: Any,
        expected: float,
        tolerance: float = 1e-10,
    ) -> int:
        try:
            return int(abs(float(value) - expected) <= tolerance)
        except (TypeError, ValueError):
            return 0

    max_length = get_value("max_length")

    checks: dict[str, int] = {
        "seed": int(get_value("seed") == 42),
        # "dataset_path": int(
        #     manifest.get("dataset_path") == "/datasets/imdb"
        # ),
        "max_length": int(max_length == 256),
    }

    if TASK_KIND == "lstm":
        checks.update({
            # model_family": int(
            #     str(
            #         get_value("model_family", "model", default="")
            #     ).lower()
            #     in {"lstm", "bilstm"}
            # ),
            "vocab_size": int(
                get_value("vocab_size") == 30000
            ),
            "embedding_dim": int(
                get_value("embedding_dim") == 128
            ),
            "hidden_dim": int(
                get_value("hidden_dim") == 128
            ),
            "bidirectional": int(
                get_value("bidirectional") is True
            ),
            "epochs": int(
                get_value("epochs") == 5
            ),
            "train_batch_size": int(
                get_value(
                    "train_batch_size",
                    "batch_size",
                ) == 128
            ),
            "learning_rate": numeric_equal(
                get_value("learning_rate"),
                1e-3,
            ),
        })

    else:
        # model_family = str(
        #     get_value("model_family", default="bert")
        # ).lower()

        # base_model = get_value(
        #     "base_model",
        #     "model",
        #     "model_name",
        # )

        checks.update({
            # "model_family": int(
            #     model_family == "bert"
            # ),
            # "base_model": int(
            #     base_model == "bert-base-uncased"
            # ),
            "epochs": int(
                get_value("epochs") == 3
            ),
            "train_batch_size": int(
                get_value(
                    "train_batch_size",
                    "batch_size",
                ) == 16
            ),
            "learning_rate": numeric_equal(
                get_value("learning_rate"),
                2e-5,
            ),
            "weight_decay": numeric_equal(
                get_value("weight_decay"),
                0.01,
            ),
            "warmup_ratio": numeric_equal(
                get_value("warmup_ratio"),
                0.1,
            ),
            # "optimizer":int(get_value("optimizer")=="AdamW"),
        })

    score = sum(checks.values()) / max(1, len(checks))

    return float(score), checks

"""def protocol_checks(manifest: dict[str, Any]) -> tuple[float, dict[str, int]]: #训练参数合规检查
    checks: dict[str, int] = {
        "seed": int(manifest.get("seed") == 42),
        "train_size": int(manifest.get("train_size") == 20000),
        "validation_size": int(manifest.get("validation_size") == 5000),
        "test_size": int(manifest.get("test_size") == 25000),
        "dataset_path": int(manifest.get("dataset_path") == "/datasets/imdb"),
        "max_length": int(manifest.get("max_length") == 256),
    }

    if TASK_KIND == "lstm":
        checks.update({
            "model_family": int(str(manifest.get("model_family", "")).lower() in {"lstm", "bilstm"}),
            "vocab_size": int(manifest.get("vocab_size") == 30000),
            "embedding_dim": int(manifest.get("embedding_dim") == 128),
            "hidden_dim": int(manifest.get("hidden_dim") == 128),
            "bidirectional": int(manifest.get("bidirectional") is True),
            "epochs": int(manifest.get("epochs") == 5),
            "train_batch_size": int(manifest.get("train_batch_size") == 128),
            "learning_rate": int(abs(float(manifest.get("learning_rate", 0.0)) - 1e-3) <= 1e-10),
        })
    else:
        checks.update({
            "model_family": int(str(manifest.get("model_family", "")).lower() == "bert"),
            "base_model": int(manifest.get("base_model") == "bert-base-uncased"),
            "epochs": int(manifest.get("epochs") == 3),
            "train_batch_size": int(manifest.get("train_batch_size") == 16),
            "learning_rate": int(abs(float(manifest.get("learning_rate", 0.0)) - 2e-5) <= 1e-10),
            "weight_decay": int(abs(float(manifest.get("weight_decay", -1.0)) - 0.01) <= 1e-10),
            "warmup_ratio": int(abs(float(manifest.get("warmup_ratio", -1.0)) - 0.1) <= 1e-10),
        })

    return sum(checks.values()) / max(1, len(checks)), checks    #同上"""


def reproducibility_score() -> float:    #可复现性评分
    paths = [SUBMISSION / "setup_env.sh", SUBMISSION / "train.py", SUBMISSION / "evaluate.py"]
    if not all(path.exists() and path.stat().st_size > 0 for path in paths):
        return 0.0
    return 1.0 if os.access(SUBMISSION / "setup_env.sh", os.X_OK) else 0.75
#os.X_OK 检查的是文件是否有chmod +x 的执行权限，这代表脚本可以直接 ./setup_env.sh 运行


def tokenize_lstm(text: str, vocab: dict[str, int]) -> tuple[list[int], int]:
    tokens = TOKEN_RE.findall(text.lower())[:MAX_LENGTH]
    ids = [int(vocab.get(token, 1)) for token in tokens] or [1]
    length = len(ids)
    ids.extend([0] * (MAX_LENGTH - len(ids)))
    return ids[:MAX_LENGTH], min(length, MAX_LENGTH)


def evaluate_lstm(rows: list[tuple[str, int, str]]):
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence

    config = load_json(SUBMISSION / "model/model_config.json", {})
    vocab = load_json(SUBMISSION / "model/vocab.json", {})
    expected = {
        "embedding_dim": 128, "hidden_dim": 128, "num_layers": 1,
        "bidirectional": True, "dropout": 0.2, "num_classes": 2, "pad_id": 0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"model_config[{key}] must equal {value!r}")

    class BiLSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(len(vocab), 128, padding_idx=0)
            self.lstm = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
            self.dropout = nn.Dropout(0.2)
            self.classifier = nn.Linear(256, 2)

        def forward(self, input_ids, lengths):
            embedded = self.embedding(input_ids)
            packed = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
            features = torch.cat((hidden[-2], hidden[-1]), dim=1)
            return self.classifier(self.dropout(features))

    checkpoint = torch.load(SUBMISSION / "model/model.pt", map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    model = BiLSTMClassifier()
    model.load_state_dict(state, strict=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    predictions, probabilities = [], []

    with torch.inference_mode():
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            encoded = [
                tokenize_lstm(text, vocab)
                for _, _, text in batch
            ]
            ids = torch.tensor([item[0] for item in encoded], device=device)
            lengths = torch.tensor([item[1] for item in encoded], device=device)
            logits = model(ids, lengths)
            probabilities.extend(float(value) for value in torch.softmax(logits, -1)[:, 1].cpu())
            predictions.extend(int(value) for value in logits.argmax(-1).cpu())

    metadata = {
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "vocab_entries": len(vocab),
    }
    return predictions, probabilities, metadata


def evaluate_bert(rows: list[tuple[str, int, str]]):  #模型推理
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = SUBMISSION / "model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    if int(model.config.num_labels) != 2:
        raise ValueError("BERT model must contain two output labels")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    predictions, probabilities = [], []

    with torch.inference_mode():
        for start in range(0, len(rows), 64):
            batch = rows[start:start + 64]
            texts = [text for _, _, text in batch]
            encoded = tokenizer(
                texts, padding=True, truncation=True, max_length=256, return_tensors="pt"
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probabilities.extend(float(value) for value in torch.softmax(logits, -1)[:, 1].cpu())
            predictions.extend(int(value) for value in logits.argmax(-1).cpu())

    metadata = {
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    return predictions, probabilities, metadata


def consistency_score(actual: dict[str, float]) -> tuple[float, dict[str, float],float]:   
    #指标一致性检查  Agent计算的测试集指标和验证器计算的测试集指标
    #预设模型主指标应达到任务预先标定的质量合格线

    baseline = load_json(BASELINE_FILE, {})
    reported = load_json(SUBMISSION / "metrics.json", {}) 
    differences = {}
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        try:
            differences[key] = abs(float(reported[key]) - actual[key])
        except Exception:
            differences[key] = 1.0

    baseline_accuracy = float(baseline.get("baseline_accuracy", 1.0))  # 基准准确率，默认1.0（没有文件时最严格）

    if actual["accuracy"] >= baseline_accuracy:
        model_quality = 1.0
    else:
        model_quality = float( abs(actual["accuracy"]-baseline_accuracy) <= 0.01 )
    
    
    return float(max(differences.values()) <= 0.01), differences,model_quality 
# 5个指标里误差最大的 ≤ 0.01 才算一致 


def ratio(numerator: int, denominator: int, default: float = 1.0) -> float:
    return default if denominator == 0 else clamp(numerator / denominator)


def canonical_arguments(arguments: Any) -> str:
    return json.dumps(
        arguments or {},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def infer_status(result: dict[str, Any]) -> str:
    extra = result.get("extra") or {}
    status = extra.get("status")
    if status:
        return str(status).lower()
    return "error" if extra.get("is_error") else "success"


def collect_calls(data: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    for step in data.get("steps") or []:
        if not isinstance(step, dict):
            continue

        results = {
            str(item.get("source_call_id")): item
            for item in ((step.get("observation") or {}).get("results") or [])
            if isinstance(item, dict) and item.get("source_call_id")
        }

        for call in step.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue

            call_id = str(call.get("tool_call_id") or "")
            result = results.get(call_id)
            arguments = call.get("arguments") or {}

            calls.append({
                "step_id": int(step.get("step_id") or 0),
                "tool_call_id": call_id,
                "tool_name": str(call.get("function_name") or "unknown"),
                "signature": (
                    str(call.get("function_name") or "unknown"),
                    canonical_arguments(arguments),
                ),
                "call_extra": call.get("extra") or {},
                "result": result,
                "result_extra": (result or {}).get("extra") or {},
                "status": infer_status(result) if result else "missing",
            })

    return calls


def calculate_process_score(
    calls: list[dict[str, Any]],
    duplicate_window: int,
) -> tuple[float, dict[str, Any]]:
    """计算执行过程质量，所有评分均为0~100。"""
    total = len(calls)

    # ============ 在这里添加幂等工具列表 ============
    IDEMPOTENT_TOOLS = {
        # 状态查询
        "list_directory",
        "check_file_exists",
        "check_directory_exists",
        "get_file_info",
        "get_current_directory",
        "get_environment_variable",
        "list_environment_variables",
        "get_system_info",
        "get_disk_usage",
        "get_file_permissions",
        # 监控场景（可选）
        "read_file",
    }

    closure_count = sum(call["result"] is not None for call in calls)
    successful_count = sum(call["status"] in SUCCESS_STATUSES for call in calls)
    intended_count = sum(
        call["call_extra"].get("intended_execution") is not False
        for call in calls
    )

    previous_by_signature: dict[tuple[str, str], dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []

    for call in calls:
        # ============ 在这里添加跳过逻辑 ============
        if call["tool_name"] in IDEMPOTENT_TOOLS:
            continue  # 跳过幂等工具的重复检测

        previous = previous_by_signature.get(call["signature"])
        if (
            previous
            and call["step_id"] - previous["step_id"] <= duplicate_window
            and previous["status"] in SUCCESS_STATUSES
        ):
            duplicates.append({
                "step_id": call["step_id"],
                "previous_step_id": previous["step_id"],
                "tool_name": call["tool_name"],
                "tool_call_id": call["tool_call_id"],
            })
        previous_by_signature[call["signature"]] = call

    closure = ratio(closure_count, total)
    execution_success = ratio(successful_count, total)
    non_duplicate = clamp(1.0 - ratio(len(duplicates), total, default=0.0))
    intended_execution = ratio(intended_count, total)

    #task_quality_1
    # score = clamp_100(100.0 * (
    #     0.40 * closure
    #     + 0.30 * execution_success
    #     # + 0.20 * non_duplicate
    #     + 0.30 * intended_execution
    # ))

    #task_quality_2
    # score = clamp_100(100.0 * (
    #         0.40 * closure
    #         + 0.30 * execution_success
    #         + 0.15 * non_duplicate
    #         + 0.15 * intended_execution
    #     ))
    
    #task_quality_3
    # score = clamp_100(100.0 * (
    #         0.25 * closure
    #         + 0.25 * execution_success
    #         + 0.25 * non_duplicate
    #         + 0.25 * intended_execution
    #     ))

    #task_quality_4
    score = clamp_100(100.0 * (
                0.20 * closure
                + 0.30 * execution_success
                + 0.20 * non_duplicate
                + 0.30 * intended_execution
            ))
    
    details = {
        "score": round(score, 2),
        "components": {
            "tool_result_closure": round(closure * 100.0, 2),
            "execution_success": round(execution_success * 100.0, 2),
            "non_duplicate_execution": round(non_duplicate * 100.0, 2),
            "intended_execution": round(intended_execution * 100.0, 2),
        },
        "counts": {
            "tool_calls": total,
            "tool_calls_with_results": closure_count,
            "successful_or_cached_results": successful_count,
            "short_range_exact_duplicates": len(duplicates),
            "unintended_calls": total - intended_count,
        },
        "duplicate_examples": duplicates[:20],
    }
    return round(score, 2), details


def calculate_robustness_score(
    calls: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """计算稳定性与异常恢复质量，所有评分均为0~100。"""
    issues: list[tuple[int, dict[str, Any]]] = []

    for index, call in enumerate(calls):
        is_error = bool(call["result_extra"].get("is_error"))
        if call["status"] in ISSUE_STATUSES or is_error:
            issues.append((index, call))

    # 没有异常、阻断、拒绝等情况时，稳定性记为100分。
    if not issues:
        return 100.0, {
            "score": 100.0,
            "evidence": "no_material_issue_observed",
            "components": {
                "recovery_rate": 100.0,
                "repeated_issue_avoidance": 100.0,
                "terminal_stability": 100.0,
            },
            "counts": {
                "issues": 0,
                "recovered_issues": 0,
                "repeated_issue_excess": 0,
            },
            "issue_details": [],
        }

    recovered_count = 0
    issue_details: list[dict[str, Any]] = []

    for index, call in issues:
        recovered = any(
            later["tool_name"] == call["tool_name"]
            and later["status"] in SUCCESS_STATUSES
            for later in calls[index + 1 :]
        )
        recovered_count += int(recovered)

        issue_details.append({
            "step_id": call["step_id"],
            "tool_call_id": call["tool_call_id"],
            "tool_name": call["tool_name"],
            "status": call["status"],
            "skip_reason": call["result_extra"].get("skip_reason"),
            "error_message": call["result_extra"].get("error_message"),
            "recovered_by_later_same_tool": recovered,
        })

    signature_counts = Counter(call["signature"] for _, call in issues)
    repeated_issue_excess = sum(
        max(0, count - 1)
        for count in signature_counts.values()
    )

    recovery_rate = ratio(recovered_count, len(issues), default=0.0)
    repeated_issue_avoidance = clamp(
        1.0 - ratio(repeated_issue_excess, len(issues), default=0.0)
    )

    last_issue_index = issues[-1][0]
    terminal_stability = 1.0 if any(
        call["status"] in SUCCESS_STATUSES
        for call in calls[last_issue_index + 1 :]
    ) else 0.0

    score = clamp_100(100.0 * (
        0.60 * recovery_rate
        + 0.25 * repeated_issue_avoidance
        + 0.15 * terminal_stability
    ))

    return round(score, 2), {
        "score": round(score, 2),
        "evidence": "material_issue_observed",
        "components": {
            "recovery_rate": round(recovery_rate * 100.0, 2),
            "repeated_issue_avoidance": round(
                repeated_issue_avoidance * 100.0, 2
            ),
            "terminal_stability": round(terminal_stability * 100.0, 2),
        },
        "counts": {
            "issues": len(issues),
            "recovered_issues": recovered_count,
            "repeated_issue_excess": repeated_issue_excess,
        },
        "issue_details": issue_details[:50],
    }

# =========================================================
# ATIF-based deterministic trajectory quality baseline
#
# Data fields follow Harbor ATIF-v1.7:
# - Step.tool_calls
# - ToolCall.tool_call_id/function_name/arguments/extra
# - Step.observation.results
# - ObservationResult.source_call_id/content/extra
#
# Harbor ATIF defines the trajectory schema and reference
# relationships, but does not define task-quality formulas.
# The following scores and weights are project-specific
# deterministic heuristics for FlagScale-Agent evaluation.
# =========================================================
def score_trajectory(
    data: dict[str, Any],
    duplicate_window: int = DUPLICATE_WINDOW,
) -> tuple[float, float, float, dict[str, Any], dict[str, Any]]:
    """返回任务质量、过程质量和稳定性评分，范围均为0~100。"""
    empty_process = {
        "score": 0.0,
        "components": {},
        "counts": {},
        "duplicate_examples": [],
    }
    empty_robustness = {
        "score": 0.0,
        "evidence": "trajectory_missing_or_empty",
        "components": {},
        "counts": {},
        "issue_details": [],
    }

    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        return 0.0, 0.0, 0.0, empty_process, empty_robustness

    calls = collect_calls(data)
    if not calls:
        return 0.0, 0.0, 0.0, empty_process, empty_robustness

    process_score, process_detail = calculate_process_score(
        calls, duplicate_window
    )
    robustness_score, robustness_detail = calculate_robustness_score(calls)

    task_quality = clamp_100(
        PROCESS_WEIGHT * process_score
        + ROBUSTNESS_WEIGHT * robustness_score
    )

    return (
        round(task_quality, 2),
        round(process_score, 2),
        round(robustness_score, 2),
        process_detail,
        robustness_detail,
    )


def main() -> int:
    VERIFIER_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_json(BASELINE_FILE, {})
    dataset_integrity, dataset_counts = check_dataset()  #数据集完整性
    artifile_score, artifact_status = artifact_checks()  #生成文件检查  模型文件等
    manifest = load_json(SUBMISSION / "run_manifest.json", {})
    reported = load_json(SUBMISSION / "metrics.json", {}) 
    data = load_json(AGENT_DIR/"trajectory.json",{})

    protocol_score, protocol_status = protocol_checks(manifest if isinstance(manifest, dict) else {}) #超参是否一致
    reproducible = reproducibility_score()  #可复现性评分

    rows = collect_rows("test")  #2000条测试数据
    labels = [label for _, label, _ in rows]
    predictions, probabilities, metadata = [], [], {}
    model_loadable = 0.0
    evaluation_error = ""

    started = time.monotonic()
    try:
        if len(rows) != 25000:
            raise ValueError(f"expected 25000 test files, found {len(rows)}")
        if TASK_KIND == "lstm":
            predictions, probabilities, metadata = evaluate_lstm(rows)
        else:
            predictions, probabilities, metadata = evaluate_bert(rows)  #加载模型
        model_loadable = 1.0   #只有完整跑完才设为1.0
    except Exception as exc:
        evaluation_error = f"{type(exc).__name__}: {exc}"
    verifier_eval_sec = time.monotonic() - started

    coverage = len(predictions) / 25000.0 if predictions else 0.0
    if model_loadable and len(predictions) == len(labels):
        metrics = compute_metrics(labels, predictions, probabilities)   #编写脚本重新计算各种指标结果
    else:
        metrics = {key: 0.0 for key in (
            "accuracy", "precision", "recall", "f1", "roc_auc", "tp", "tn", "fp", "fn"
        )}

    consistent, differences , model_quality_pass= consistency_score(metrics) #计算机指标
    runtime = load_json(SUBMISSION / "runtime.json", {})
    device_text = str(runtime.get("device", metadata.get("device", ""))).lower() if isinstance(runtime, dict) else ""
    peak_memory = float(runtime.get("peak_gpu_memory_mb", 0.0) or 0.0) if isinstance(runtime, dict) else 0.0
    gpu_used = float("cuda" in device_text and peak_memory > 0.0)

    baseline_accuracy = float(baseline.get("baseline_accuracy", 1.0))  # 基准准确率，默认1.0（没有文件时最严格）
    baseline_f1 = float(baseline.get("baseline_f1", baseline_accuracy))  # 基准F1，没有就用baseline_accuracy 代替
    baseline_auc = float(baseline.get("baseline_roc_auc", 1.0)) # 基准ROC AUC，默认1.0

    #任务质量评分
    task_quality, process_quality, robustness_quality,process_detail ,robustness_detail = score_trajectory(data)

    task_success = float(
            artifile_score == 1.0 and
            protocol_score == 1.0 and
            consistent == 1.0 and
            model_loadable == 1.0 and
            model_quality_pass == 1.0
        )
    # RewardKit programmatic rewards use [0, 1].
    programmatic_scores = {
        "correct": task_success,
        "assistant_quality": round(task_quality / 100.0, 6),
    }
    save_json(VERIFIER_DIR / "programmatic-scores.json", programmatic_scores)

    report = {
        "task_kind": TASK_KIND,
        "programmatic_scores": programmatic_scores,
        # "task_success": task_success,
        # "assistant_quality_100": task_quality,
        "process_quality_100": process_quality,
        "robustness_quality_100": robustness_quality,
        "completion_checks": {
            # "dataset_integrity": dataset_integrity,
            "artifact_completeness": artifile_score,
            "protocol_compliance": protocol_score,
            "reproducible_setup": reproducible,
            "model_loadable": model_loadable,
            "evaluation_test_coverage": clamp(coverage),
            "metrics_consistent": consistent,
            "model_quality_pass": model_quality_pass,
            "gpu_used": gpu_used,
        },
        "metrics": {
            "baseline": {
                "accuracy": baseline_accuracy,
                "f1": baseline_f1,
                "roc_auc": baseline_auc,
            },
            "verifier": {
                key: float(metrics[key])
                for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
            },
            "agent": {
                key: float(reported.get(key, 0.0) or 0.0)
                for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
            },
        },
        "verifier_eval_time": verifier_eval_sec,
        "peak_gpu_memory_mb": peak_memory,
        "confusion_matrix": {
            key: int(metrics[key])
            for key in ("tp", "tn", "fp", "fn")
        },
        "process_detail": process_detail,
        "robustness_detail": robustness_detail,
        "dataset_counts": dataset_counts,
        "artifact_checks": artifact_status,
        "protocol_checks": protocol_status,
        "reported_metric_differences": differences,
        "model_metadata": metadata,
        "evaluation_error": evaluation_error,
        "runtime_reported_by_agent": runtime,
    }
    save_json(ARTIFACT_DIR / "evaluation_report.json", report)

    if len(predictions) == len(rows):
        with (ARTIFACT_DIR / "verifier_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for (sample_id, label, _), prediction, probability in zip(rows, predictions, probabilities):
                handle.write(json.dumps({
                    "id": sample_id, "label": label, "prediction": prediction,
                    "probability": probability,
                }) + "\n")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
