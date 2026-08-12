# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Anthropic provider implementation."""

import json

from typing import Any, Dict, Iterator, List

import anthropic

from flagscale_agent.react.providers.base import LLMProvider

import time
import uuid

from flagscale_agent.trace_logger import trace_logger

class AnthropicProvider(LLMProvider):
    schema_format = "anthropic"

    def __init__(self, model: str, api_key: str, base_url: str = None, max_tokens: int = 8192):
        self._model = model
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._base_url = base_url
        self._is_third_party = base_url and "anthropic.com" not in base_url
        self._auth_mode = None  # Will be auto-detected on first call
        self._timeout = 120.0  # 2-minute timeout for API calls + summarizer
        self._client = self._build_client()

    def _build_client(self):
        """Build Anthropic client with current auth mode."""
        kwargs = {"api_key": self._api_key, "timeout": self._timeout}
        if self._base_url:
            kwargs["base_url"] = self._base_url
            if self._is_third_party and self._auth_mode == "bearer":
                kwargs["api_key"] = "placeholder"
                kwargs["default_headers"] = {"Authorization": f"Bearer {self._api_key}"}
        return anthropic.Anthropic(**kwargs)

    def _switch_auth_and_retry(self):
        """Switch from x-api-key to Bearer auth after a 401."""
        if self._auth_mode == "bearer":
            return False  # Already tried Bearer, nothing more to do
        self._auth_mode = "bearer"
        self._client = self._build_client()
        return True

    def _split_system(self, messages):
        """Separate system message from chat messages (Anthropic requires this)."""
        system = None
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)
        return system, chat_messages

    def _build_kwargs(self, messages, tools):
        system, chat_messages = self._split_system(messages)
        kwargs = {"model": self._model, "max_tokens": self._max_tokens, "messages": chat_messages}
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        return kwargs

    @staticmethod
    def _usage_to_dict(usage: Any) -> Dict[str, Any]:
        """Convert Anthropic usage object to a JSON-friendly dict."""
        if usage is None:
            return {}

        return {
            "input_tokens": int(
                getattr(usage, "input_tokens", 0) or 0
            ),
            "output_tokens": int(
                getattr(usage, "output_tokens", 0) or 0
            ),
            "cache_creation_input_tokens": int(
                getattr(
                    usage,
                    "cache_creation_input_tokens",
                    0,
                )
                or 0
            ),
            "cache_read_input_tokens": int(
                getattr(
                    usage,
                    "cache_read_input_tokens",
                    0,
                )
                or 0
            ),
        }
    
    # def chat(self, messages: List[Dict[str, Any]], tools: List[dict]) -> Dict[str, Any]:
        kwargs = self._build_kwargs(messages, tools)
        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError:
            if self._is_third_party and self._switch_auth_and_retry():
                response = self._client.messages.create(**kwargs)
            else:
                raise

        content = None
        tool_calls = None
        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({"id": block.id, "name": block.name, "arguments": block.input})

        return {"content": content, "tool_calls": tool_calls}
    def chat(self,messages: List[Dict[str, Any]],tools: List[dict],) -> Dict[str, Any]:
        """Non-streaming Anthropic request with structured tracing."""

        kwargs = self._build_kwargs(messages, tools)

        call_id = f"llm_{uuid.uuid4().hex}"
        started_at = time.monotonic()

        # 记录本轮真正发给模型的完整输入
        trace_logger.emit(
            "llm_request",
            call_id=call_id,
            provider="anthropic",
            model=self._model,
            auth_mode=self._auth_mode,
            request={
                "system": kwargs.get("system"),
                "messages": kwargs.get("messages", []),
                "tools": kwargs.get("tools", []),
                "max_tokens": kwargs.get("max_tokens"),
            },
        )

        try:
            try:
                response = self._client.messages.create(
                    **kwargs
                )

            except anthropic.AuthenticationError as exc:
                if (
                    self._is_third_party
                    and self._switch_auth_and_retry()
                ):
                    trace_logger.emit(
                        "llm_retry",
                        call_id=call_id,
                        provider="anthropic",
                        model=self._model,
                        reason="authentication_error",
                        previous_auth_mode="x-api-key",
                        new_auth_mode="bearer",
                        error_type=type(exc).__name__,
                    )

                    response = self._client.messages.create(
                        **kwargs
                    )
                else:
                    raise

        except Exception as exc:
            trace_logger.emit(
                "llm_error",
                call_id=call_id,
                provider="anthropic",
                model=self._model,
                error_type=type(exc).__name__,
                error_message=str(exc),
                latency_sec=(
                    time.monotonic() - started_at
                ),
            )
            raise

        content = None
        tool_calls = None

        for block in response.content:
            if block.type == "text":
                # 保持你原来的返回行为
                content = block.text

            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []

                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    }
                )

        usage = self._usage_to_dict(
            getattr(response, "usage", None)
        )

        # 记录Anthropic返回的完整公开响应
        trace_logger.emit(
            "llm_response",
            call_id=call_id,
            provider="anthropic",
            model=getattr(
                response,
                "model",
                self._model,
            ),
            response=response,
            parsed_content=content,
            parsed_tool_calls=tool_calls,
            stop_reason=getattr(
                response,
                "stop_reason",
                None,
            ),
            stop_sequence=getattr(
                response,
                "stop_sequence",
                None,
            ),
            usage=usage,
            latency_sec=(
                time.monotonic() - started_at
            ),
        )

        return {
            "content": content,
            "tool_calls": tool_calls,
        }

    # def chat_stream(self, messages: List[Dict[str, Any]], tools: List[dict]) -> Iterator[Dict[str, Any]]:
        kwargs = self._build_kwargs(messages, tools)
        stream_ctx = None

        try:
            stream_ctx = self._client.messages.stream(**kwargs)
            stream = stream_ctx.__enter__()
        except anthropic.AuthenticationError:
            if self._is_third_party and self._switch_auth_and_retry():
                # Close old context before creating new one
                if stream_ctx is not None:
                    try:
                        stream_ctx.__exit__(None, None, None)
                    except Exception:
                        pass
                stream_ctx = self._client.messages.stream(**kwargs)
                stream = stream_ctx.__enter__()
            else:
                raise

        stream_error = None
        try:
            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        yield {"type": "tool_start", "id": block.id, "name": block.name}
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield {"type": "text", "content": delta.text}
                    elif delta.type == "input_json_delta":
                        yield {"type": "tool_delta", "id": "", "arguments_delta": delta.partial_json}
        except Exception as e:
            stream_error = e
        finally:
            if stream_ctx is not None:
                try:
                    stream_ctx.__exit__(None, None, None)
                except Exception:
                    pass

        if stream_error:
            raise stream_error

        # Only try to get usage if stream completed normally
        if stream_ctx is not None:
            try:
                final = stream.get_final_message()
                if final and final.usage:
                    yield {
                        "type": "usage",
                        "input_tokens": final.usage.input_tokens,
                        "output_tokens": final.usage.output_tokens,
                    }
            except Exception:
                pass

        yield {"type": "done"}
    def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: List[dict],
    ) -> Iterator[Dict[str, Any]]:
        """Streaming Anthropic request with structured tracing."""

        kwargs = self._build_kwargs(messages, tools)

        call_id = f"llm_{uuid.uuid4().hex}"
        started_at = time.monotonic()

        trace_logger.emit(
            "llm_request",
            call_id=call_id,
            provider="anthropic",
            model=self._model,
            auth_mode=self._auth_mode,
            streaming=True,
            request={
                "system": kwargs.get("system"),
                "messages": kwargs.get("messages", []),
                "tools": kwargs.get("tools", []),
                "max_tokens": kwargs.get("max_tokens"),
            },
        )

        stream_ctx = None
        stream = None

        try:
            try:
                stream_ctx = self._client.messages.stream(
                    **kwargs
                )
                stream = stream_ctx.__enter__()

            except anthropic.AuthenticationError as exc:
                if (
                    self._is_third_party
                    and self._switch_auth_and_retry()
                ):
                    trace_logger.emit(
                        "llm_retry",
                        call_id=call_id,
                        provider="anthropic",
                        model=self._model,
                        reason="authentication_error",
                        previous_auth_mode="x-api-key",
                        new_auth_mode="bearer",
                        error_type=type(exc).__name__,
                    )

                    if stream_ctx is not None:
                        try:
                            stream_ctx.__exit__(
                                None,
                                None,
                                None,
                            )
                        except Exception:
                            pass

                    stream_ctx = (
                        self._client.messages.stream(
                            **kwargs
                        )
                    )
                    stream = stream_ctx.__enter__()
                else:
                    raise

            text_chunks: List[str] = []
            tool_blocks: Dict[int, Dict[str, Any]] = {}

            final_message = None

            for event in stream:
                event_type = getattr(
                    event,
                    "type",
                    "",
                )
                event_index = int(
                    getattr(event, "index", -1)
                )

                if event_type == "content_block_start":
                    block = event.content_block

                    if block.type == "tool_use":
                        tool_blocks[event_index] = {
                            "id": block.id,
                            "name": block.name,
                            "arguments_json": "",
                        }

                        yield {
                            "type": "tool_start",
                            "id": block.id,
                            "name": block.name,
                        }

                elif event_type == "content_block_delta":
                    delta = event.delta

                    if delta.type == "text_delta":
                        text_chunks.append(delta.text)

                        yield {
                            "type": "text",
                            "content": delta.text,
                        }

                    elif delta.type == "input_json_delta":
                        partial_json = (
                            delta.partial_json or ""
                        )

                        if event_index in tool_blocks:
                            tool_blocks[event_index][
                                "arguments_json"
                            ] += partial_json

                        yield {
                            "type": "tool_delta",
                            "id": (
                                tool_blocks
                                .get(event_index, {})
                                .get("id", "")
                            ),
                            "arguments_delta": partial_json,
                        }

            # 在关闭stream前读取最终完整Message
            try:
                final_message = (
                    stream.get_final_message()
                )
            except Exception as exc:
                trace_logger.emit(
                    "llm_final_message_error",
                    call_id=call_id,
                    provider="anthropic",
                    model=self._model,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

        except Exception as exc:
            trace_logger.emit(
                "llm_error",
                call_id=call_id,
                provider="anthropic",
                model=self._model,
                streaming=True,
                error_type=type(exc).__name__,
                error_message=str(exc),
                latency_sec=(
                    time.monotonic() - started_at
                ),
            )
            raise

        finally:
            if stream_ctx is not None:
                try:
                    stream_ctx.__exit__(
                        None,
                        None,
                        None,
                    )
                except Exception:
                    pass

        usage: Dict[str, Any] = {}

        if final_message is not None:
            usage = self._usage_to_dict(
                getattr(
                    final_message,
                    "usage",
                    None,
                )
            )

            trace_logger.emit(
                "llm_response",
                call_id=call_id,
                provider="anthropic",
                model=getattr(
                    final_message,
                    "model",
                    self._model,
                ),
                streaming=True,
                response=final_message,
                stop_reason=getattr(
                    final_message,
                    "stop_reason",
                    None,
                ),
                usage=usage,
                latency_sec=(
                    time.monotonic() - started_at
                ),
            )

        else:
            # 即使SDK未返回最终Message，也保存已经聚合的信息
            parsed_tool_calls = []

            for block in tool_blocks.values():
                raw_arguments = block.get(
                    "arguments_json",
                    "",
                )

                try:
                    arguments = (
                        json.loads(raw_arguments)
                        if raw_arguments
                        else {}
                    )
                except json.JSONDecodeError:
                    arguments = {
                        "_raw_partial_json": (
                            raw_arguments
                        )
                    }

                parsed_tool_calls.append(
                    {
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "arguments": arguments,
                    }
                )

            trace_logger.emit(
                "llm_response",
                call_id=call_id,
                provider="anthropic",
                model=self._model,
                streaming=True,
                response_unavailable=True,
                parsed_content="".join(text_chunks),
                parsed_tool_calls=parsed_tool_calls,
                usage={},
                latency_sec=(
                    time.monotonic() - started_at
                ),
            )

        if usage:
            yield {
                "type": "usage",
                "input_tokens": usage.get(
                    "input_tokens",
                    0,
                ),
                "output_tokens": usage.get(
                    "output_tokens",
                    0,
                ),
                "cache_creation_input_tokens": (
                    usage.get(
                        "cache_creation_input_tokens",
                        0,
                    )
                ),
                "cache_read_input_tokens": (
                    usage.get(
                        "cache_read_input_tokens",
                        0,
                    )
                ),
            }

        yield {"type": "done"}

    def format_assistant_message(self, response: Dict[str, Any]) -> Dict[str, Any]:
        content_blocks = []
        if response["content"]:
            content_blocks.append({"type": "text", "text": response["content"]})
        if response["tool_calls"]:
            for tc in response["tool_calls"]:
                content_blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})
        if not content_blocks:
            content_blocks.append({"type": "text", "text": ""})
        return {"role": "assistant", "content": content_blocks}

    # def format_tool_result(self, tool_call_id: str, content: str) -> Dict[str, Any]:
    #     return {
    #         "role": "user",
    #         "content": [{"type": "tool_result", "tool_use_id": tool_call_id, "content": content or "(empty)"}],
    #     }

    def format_tool_result(
        self,
        tool_call_id: str,
        content: str,
    ) -> Dict[str, Any]:
        normalized_content = content or "(empty)"

        trace_logger.emit(
            "tool_result_message",
            tool_call_id=tool_call_id,
            provider="anthropic",
            content=normalized_content,
        )

        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": normalized_content,
                }
            ],
        }