#!/usr/bin/env python3

from __future__ import annotations

import sys

import litellm


def register_gateway_model() -> None:
    # 你的网关中：
    # mcs-5 -> claude-sonnet-4-6
    #
    # RewardKit 会在真正请求前调用 LiteLLM 获取模型信息，
    # 因此给 mcs-5 注册 claude-sonnet-4-6 的模型规格。
    base_info = dict(
        litellm.get_model_info(
            "anthropic/claude-sonnet-4-6"
        )
    )

    base_info["litellm_provider"] = "anthropic"
    base_info["mode"] = "chat"

    litellm.register_model(
        {
            "anthropic/mcs-5": base_info,
        }
    )

    # 提前验证，注册失败就立即报错。
    info = litellm.get_model_info("anthropic/mcs-5")

    print(
        "[rewardkit-wrapper] registered anthropic/mcs-5",
        {
            "max_input_tokens": info.get("max_input_tokens"),
            "max_output_tokens": info.get("max_output_tokens"),
            "litellm_provider": info.get("litellm_provider"),
            "mode": info.get("mode"),
        },
        file=sys.stderr,
    )


def main() -> None:
    register_gateway_model()

    # 必须在同一个 Python 进程中启动 RewardKit，
    # 否则上面的 register_model() 注册会丢失。
    from rewardkit.__main__ import main as rewardkit_main

    result = rewardkit_main()

    if isinstance(result, int):
        raise SystemExit(result)


if __name__ == "__main__":
    main()