# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import copy
import json
import logging
import os
import re
import time
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import add_generation_prompt_for_gpt_oss, format_gpt_oss_tool_response_manually
from verl.interactions.base import BaseInteraction
from verl.tools.schemas import AgentToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def write_llm_io_trace(record: dict[str, Any]) -> None:
    trace_path = os.getenv("COAGENTIC_RETRIEVER_LLM_IO_JSONL", "")
    if not trace_path:
        return
    try:
        max_records = int(os.getenv("COAGENTIC_RETRIEVER_LLM_IO_MAX_RECORDS", "0") or 0)
        if max_records > 0 and os.path.exists(trace_path):
            with open(trace_path, "r", encoding="utf-8") as fp:
                if sum(1 for _ in fp) >= max_records:
                    return
        os.makedirs(os.path.dirname(trace_path), exist_ok=True)
        with open(trace_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps({"ts": time.time(), **record}, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.warning(f"failed to write LLM IO trace: {exc}")


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
        initial_query: Optional[str] = None,
        answers: Optional[list[str]] = None
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}
        self.initial_query = initial_query
        self.answers = answers

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        # Extra fields for dynamic addition
        self.extra_fields: dict[str, Any] = {}

        self.json_correct: bool = True 
        self.one_tool_call_per_assistant: bool = True

        # Accumulates details of each tool call for ranker contrastive training.
        self.tool_call_details: list[dict[str, Any]] = []
        self.tool_call_count: int = 0
        self.has_search_tool_call: bool = False
        self.invalid_direct_answer_before_search: bool = False


@register("coagentic_retriever_agent")
class CoAgenticRetrieverAgentLoop(AgentLoopBase):
    def __init__(self, trainer_config, server_manager, tokenizer, processor, **kwargs):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level CoAgenticRetrieverAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        print(f"\n[COAGENTIC RETRIEVER AGENT] Tool config path: {tool_config_path}")
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        cls.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format
        print(f"[COAGENTIC RETRIEVER AGENT] Initialized tools: {list(cls.tools.keys())}")
        print(f"[COAGENTIC RETRIEVER AGENT] Tool instances: {cls.tools}\n")

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], tools=cls.tool_schemas, add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )

        # check tools are only for search 
        assert len(tool_list) == 1, "CoAgenticRetrieverAgentLoop only supports one tool for search."
        assert tool_list[0].name == "search", "The only supported tool is 'search'."
        assert cls.tool_schemas, "CoAgenticRetrieverAgentLoop requires tool schemas for Qwen tool formatting."

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})
        initial_query = kwargs["extra_info"]["question"]
        answers = list(kwargs["reward_model"]["ground_truth"]["target"])
        
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            initial_query=initial_query,
            answers=answers,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=None,
            interaction_kwargs={},
        )
        agent_data.extra_fields["uid"] = kwargs["uid"] # for GRPO

        # State machine loop
        worker_id = os.getpid()
        state = AgentState.PENDING
        iteration = 0
        while state != AgentState.TERMINATED:
            iteration += 1
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data, sampling_params)
            else:
                logger.error(f"[W{worker_id}] Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields=agent_data.extra_fields,
        )

        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        output.extra_fields.update({
            "json_correct": agent_data.json_correct,
            "one_tool_call_per_assistant": agent_data.one_tool_call_per_assistant,
            "tool_call_count": agent_data.tool_call_count,
            "has_search_tool_call": agent_data.has_search_tool_call,
            "min_one_search": agent_data.has_search_tool_call,
            "invalid_direct_answer_before_search": agent_data.invalid_direct_answer_before_search,
        })

        # Store tool call details so ranker sample construction can read them.
        if agent_data.tool_call_details:
            output.extra_fields["tool_call_details"] = agent_data.tool_call_details
            output.extra_fields["messages"] = agent_data.messages
            output.extra_fields["initial_query"] = agent_data.initial_query
            output.extra_fields["answers"] = agent_data.answers

        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []
        worker_id = os.getpid()
        prompt_ids_before_generate = list(agent_data.prompt_ids)

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
            )

        write_llm_io_trace(
            {
                "source": "train",
                "role": "agent",
                "pid": os.getpid(),
                "request_id": agent_data.request_id,
                "initial_query": agent_data.initial_query,
                "assistant_turn": agent_data.assistant_turns + 1,
                "user_turn": agent_data.user_turns,
                "prompt_token_count": len(prompt_ids_before_generate),
                "output_token_count": len(output.token_ids),
                "prompt_text": self.tokenizer.decode(prompt_ids_before_generate),
                "output_text": self.tokenizer.decode(output.token_ids),
                "sampling_params": sampling_params,
            }
        )
        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs
            
        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
        num_calls = len(agent_data.tool_calls) if agent_data.tool_calls else 0

        # Determine next state, we only allow 1 tool call per generation turn
        if agent_data.tool_calls and len(agent_data.tool_calls) == 1:
            self._truncate_after_first_tool_call(agent_data)
            return AgentState.PROCESSING_TOOLS

        # Check whether we get the answer only after no valid tool call was found.
        # Qwen3-0.6B can append a premature <answer> after a valid <tool_call>;
        # CoAgenticRetriever should still execute the search in that case.
        if self.detect_answer(agent_data.response_ids):
            if not agent_data.has_search_tool_call:
                agent_data.json_correct = False
                agent_data.one_tool_call_per_assistant = False
                agent_data.invalid_direct_answer_before_search = True
                logger.warning(
                    f"[W{worker_id}-GEN] final answer before any successful search/tool_call, marking invalid"
                )
            return AgentState.TERMINATED
        else:
            agent_data.one_tool_call_per_assistant = False
            logger.warning(f"[W{worker_id}-GEN] no valid tools (got {num_calls}), terminating")
            return AgentState.TERMINATED

    def _truncate_after_first_tool_call(self, agent_data: AgentData) -> None:
        """Drop tokens generated after the first closing tool-call tag."""
        end_ids = self.tokenizer.encode("</tool_call>", add_special_tokens=False)
        if not end_ids:
            return

        response_ids = agent_data.response_ids
        cut = None
        for i in range(0, len(response_ids) - len(end_ids) + 1):
            if response_ids[i : i + len(end_ids)] == end_ids:
                cut = i + len(end_ids)
                break

        if cut is None or cut >= len(response_ids):
            return

        trim = len(response_ids) - cut
        agent_data.response_ids = response_ids[:cut]
        agent_data.prompt_ids = agent_data.prompt_ids[:-trim]
        agent_data.response_mask = agent_data.response_mask[:-trim]
        if agent_data.response_logprobs:
            agent_data.response_logprobs = agent_data.response_logprobs[:-trim]

    async def _handle_processing_tools_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []

        tools_kwargs_with_context = copy.deepcopy(agent_data.tools_kwargs)
        
        for tool_name in self.tools.keys():
            if tool_name not in tools_kwargs_with_context:
                tools_kwargs_with_context[tool_name] = {}
            
            if "create_kwargs" not in tools_kwargs_with_context[tool_name]:
                tools_kwargs_with_context[tool_name]["create_kwargs"] = {}
            
            create_kwargs = tools_kwargs_with_context[tool_name]["create_kwargs"]
            create_kwargs["request_id"] = agent_data.request_id
            create_kwargs["initial_query"] = agent_data.initial_query
            create_kwargs["answers"] = agent_data.answers
        assert len(agent_data.tool_calls) == 1, "CoAgenticRetrieverAgentLoop only supports one tool call at a time."

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, tools_kwargs_with_context, agent_data=agent_data))
            tool_call_names.append(tool_call.name)
   
        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)
        
        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_call_name, (tool_response, tool_reward, _) in zip(tool_call_names, responses):
            message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

            if "Error when executing tool:" in message["content"]:
                agent_data.json_correct = False
            else:
                agent_data.tool_call_count += 1
                if tool_call_name == "search":
                    agent_data.has_search_tool_call = True

        assert len(add_messages) == 1, "CoAgenticRetrieverAgentLoop only supports one tool call at a time."
        agent_data.messages.extend(add_messages)
        # Update prompt with tool responses
        if self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    tools=self.tool_schemas,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_tool_response], images=None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if self.tool_parser_name == "gpt-oss":
                logger.info("manually format tool responses for gpt-oss")
                # Format tool responses manually
                tool_response_texts = []
                for i, tool_msg in enumerate(add_messages):
                    actual_tool_name = tool_call_names[i]
                    formatted = format_gpt_oss_tool_response_manually(tool_msg["content"], actual_tool_name)
                    tool_response_texts.append(formatted)

                tool_response_text = add_generation_prompt_for_gpt_oss("".join(tool_response_texts))
                response_ids = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
                )
            else:
                response_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(
                        add_messages,
                        tools=self.tool_schemas,
                        add_generation_prompt=True,
                        tokenize=True,
                        **self.apply_chat_template_kwargs,
                    ),
                )
                response_ids = response_ids[len(self.system_prompt) :]
        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1

        return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData,
    ) -> tuple[AgentToolResponse, float, dict]:
        """Call tool and return tool response."""
        worker_id = os.getpid()
        tool, instance_id = None, None
        tool_name = tool_call.name
        try:
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.warning(f"[W{worker_id}-TOOL-{tool_name}] ERROR: {e}")
            return (
                AgentToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {"ranker_failed": True},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        full_tool_response_text = tool_execution_response.text or ""
        tool_response_text = full_tool_response_text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_execution_response.text = tool_response_text
        
        # Collect tool call details for ranker contrastive training.
        tool_call_detail = {
            "step_index": len(agent_data.tool_call_details),
            "sub_query": res.get("sub_query", ""),
            "tool_score": res.get("tool_score", 0.0),
            "answer_in_docs": res.get("answer_in_docs", False),
            "ranker_failed": res.get("ranker_failed", False),
            "full_observation": full_tool_response_text,
            "agent_observation": tool_response_text,
            "observation_truncated": full_tool_response_text != tool_response_text,
            "full_observation_chars": len(full_tool_response_text),
            "agent_observation_chars": len(tool_response_text or ""),
            "max_tool_response_length": self.max_tool_response_length,
            "tool_response_truncate_side": self.tool_response_truncate_side,
        }
        if "recall_top50_docs" in res:
            tool_call_detail["recall_top50_docs"] = res["recall_top50_docs"]
        if "rank_top50_docs" in res:
            tool_call_detail["rank_top50_docs"] = res["rank_top50_docs"]
        if "rank_top5_docs" in res:
            tool_call_detail["rank_top5_docs"] = res["rank_top5_docs"]
        agent_data.tool_call_details.append(tool_call_detail)

        return tool_execution_response, tool_reward, res

    def detect_answer(self, response_ids: list[int]) -> bool:
        """Detect if the model has provided a final answer in its response.

        The function looks for the presence of an <answer>...</answer> tag
        in the decoded text of the response IDs.

        Example valid output:
        <reason>...</reason>
        <answer>...</answer>

        Args:
            response_ids: The list of token IDs generated by the model.
        Returns:
            True if a final answer is detected, False otherwise.
        """
        text = self.tokenizer.decode(response_ids)
        # Use strict regex to match <answer>...</answer>
        answer_regex = r"<answer>(.*?)</answer>"
        match = re.search(answer_regex, text, re.DOTALL)
        return match is not None
