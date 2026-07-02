"""CoAgenticRetriever trainer with trainable E5 ranker contrastive steps."""

from __future__ import annotations

import contextlib
import os
import threading
import uuid
from pprint import pprint

import numpy as np
import ray
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.coagentic_retriever_ray_trainer import (
    CoAgenticRetrieverRayTrainer,
    compute_response_mask,
    process_main_agent_ppo_step,
)
from verl.trainer.ppo.ranker_contrastive_step import process_ranker_contrastive_step
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer

from ranker_strategies.config import (
    build_collator,
    build_construction_logger,
    build_replay_buffer,
    build_sample_builder,
    build_selector,
    build_signal_builder,
)
from ranker_strategies.trajectory_selector import build_fresh_trajectories_from_dataproto
from verl.workers.ranker.e5_ranker_worker import LocalRankerContrastiveWorker, SharedE5RankerActor
from async_ranker_training.strategies.config import build_async_trajectory_selector
from async_ranker_training.config import load_async_ranker_training_config
from async_ranker_training.labeler import AsyncLabeler
from async_ranker_training.ranker_async_trainer import RankerAsyncTrainer
from async_ranker_training.request_builder import build_requests_from_contexts


def _cfg_get(config, path: str, default=None):
    cur = config
    for part in path.split("."):
        if hasattr(cur, "get"):
            cur = cur.get(part, default)
        elif isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            return default
        if cur is default:
            return default
    return cur


def _cfg_require(config, path: str):
    sentinel = object()
    value = _cfg_get(config, path, sentinel)
    if value is sentinel or value is None or value == "":
        raise KeyError(f"missing required ranker config: {path}")
    return value


def _cfg_require_present(config, path: str):
    sentinel = object()
    value = _cfg_get(config, path, sentinel)
    if value is sentinel:
        raise KeyError(f"missing required ranker config: {path}")
    return value


class _LocalRankerWG:
    """Small adapter exposing the same method as a Ray worker group."""

    def __init__(self, config):
        self.worker = LocalRankerContrastiveWorker(config)

    def init_model(self):
        return self.worker.init_model()

    def update_ranker_contrastive(self, batch):
        return self.worker.update_ranker_contrastive(batch)

    def save_checkpoint(self, path: str):
        return self.worker.save_checkpoint(path)

    def export_encoder_state_cpu(self):
        return self.worker.export_encoder_state_cpu()


def _as_python_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class CoAgenticRankerContrastiveRayTrainer(CoAgenticRetrieverRayTrainer):
    """Run agent PPO/GRPO and ranker InfoNCE updates in one training loop."""

    def init_workers(self):
        super().init_workers()
        if not self._ranker_train_enabled():
            self.ranker_wg = None
            return
        self._init_ranker_components()

    def _ranker_train_enabled(self) -> bool:
        return bool(_cfg_require(self.config, "trainer.ranker_trainable")) and (
            _cfg_require(self.config, "trainer.ranker_update_mode") == "contrastive"
        )

    def _should_dump_rollout_data(self, is_last_step: bool) -> bool:
        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
        if not rollout_data_dir:
            return False

        dump_every = int(self.config.trainer.get("dump_rollout_every_step_num", 10))
        if dump_every <= 0:
            return False
        return is_last_step or self.global_steps % dump_every == 0

    def _limit_rollout_dump_batch(self, batch: DataProto) -> DataProto:
        dump_num = int(self.config.trainer.get("dump_rollout_num_everytime", 1))
        if dump_num < 0 or dump_num >= len(batch):
            return batch
        if dump_num == 0:
            return batch[:0]
        return batch[:dump_num]

    def _limit_rollout_dump_reward_infos(self, reward_extra_infos_dict: dict, dump_size: int) -> dict:
        limited_infos = {}
        for key, value in (reward_extra_infos_dict or {}).items():
            try:
                if len(value) >= dump_size:
                    limited_infos[key] = value[:dump_size]
                    continue
            except TypeError:
                pass
            limited_infos[key] = value
        return limited_infos

    def _prune_rollout_dump_files(self, rollout_dump_dir: str):
        max_dump_num = int(self.config.trainer.get("max_rollout_dump_num", -1))
        if max_dump_num < 0 or not os.path.isdir(rollout_dump_dir):
            return

        files = []
        for name in os.listdir(rollout_dump_dir):
            if not name.endswith(".jsonl"):
                continue
            stem = name[:-6]
            if not stem.isdigit():
                continue
            files.append((int(stem), os.path.join(rollout_dump_dir, name)))

        files.sort(key=lambda item: item[0])
        extra = len(files) - max_dump_num
        if extra <= 0:
            return
        for _, path in files[:extra]:
            with contextlib.suppress(FileNotFoundError):
                os.remove(path)

    def _init_ranker_components(self):
        self.ranker_wg = _LocalRankerWG(self.config)
        self.ranker_wg.init_model()
        self.ranker_lock = threading.RLock()
        self.ranker_replay_buffer = build_replay_buffer(self.config)
        self.ranker_selector = build_selector(self.config)
        self.ranker_signal_builder = build_signal_builder(self.config)
        self.ranker_sample_builder = build_sample_builder(self.config)
        self.ranker_collator = build_collator(self.config, self.ranker_wg.worker.tokenizer)
        self.ranker_construction_logger = build_construction_logger(self.config)
        self.ranker_update_step = 0
        self.ranker_async_last_logged_updates = 0
        self.ranker_inference_actor = None
        self.ranker_inference_actor_name = self._ranker_inference_actor_name()
        self.ranker_inference_last_sync_step = -1
        if self._shared_inference_ranker_enabled():
            self._init_shared_inference_ranker()
        self.async_ranker_training_config = load_async_ranker_training_config(self.config)
        self.async_trajectory_selector = None
        self.async_ranker_labeler = None
        self.ranker_async_trainer = None
        if self._async_ranker_training_enabled():
            self.async_trajectory_selector = build_async_trajectory_selector(
                self.async_ranker_training_config.trajectory_selector,
                trainer_config=self.config,
            )
            log_dir = self._async_ranker_training_log_dir()
            self.async_ranker_labeler = AsyncLabeler(
                self.async_ranker_training_config,
                project_root=os.getcwd(),
                log_dir=log_dir,
            )
            self.async_ranker_labeler.start()
            self.ranker_async_trainer = RankerAsyncTrainer(
                config=self.config,
                async_config=self.async_ranker_training_config,
                async_ranker_labeler=self.async_ranker_labeler,
                ranker_wg=self.ranker_wg,
                replay_buffer=self.ranker_replay_buffer,
                collator=self.ranker_collator,
                construction_logger=self.ranker_construction_logger,
                ranker_lock=self.ranker_lock,
                after_update_callback=self._maybe_sync_shared_inference_ranker,
            )
            if bool(_cfg_require(self.config, "ranker_training.async_ranker_training.background_ranker_thread")):
                self.ranker_async_trainer.start()

    def _shared_inference_ranker_enabled(self) -> bool:
        return bool(_cfg_require(self.config, "ranker_training.shared_inference_ranker.enable"))

    def _ranker_inference_actor_name(self) -> str:
        return str(_cfg_require(self.config, "ranker_training.shared_inference_ranker.actor_name"))

    def _ranker_inference_actor_namespace(self):
        return _cfg_require_present(self.config, "ranker_training.shared_inference_ranker.actor_namespace")

    def _ranker_inference_sync_interval(self) -> int:
        return int(_cfg_require(self.config, "ranker_training.shared_inference_ranker.sync_interval"))

    def _init_shared_inference_ranker(self):
        ranker_config = OmegaConf.to_container(self.config.ranker, resolve=True)
        ranker_config["backend"] = "ray_actor"
        ranker_config["actor_name"] = self.ranker_inference_actor_name
        actor_config = OmegaConf.to_container(self.config, resolve=True)
        actor_config["ranker"] = ranker_config
        namespace = self._ranker_inference_actor_namespace()
        actor_options = {
            "name": self.ranker_inference_actor_name,
            "get_if_exists": True,
        }
        if namespace:
            actor_options["namespace"] = str(namespace)
        self.ranker_inference_actor = SharedE5RankerActor.options(**actor_options).remote(actor_config)
        status = ray.get(self.ranker_inference_actor.get_status.remote())
        print(
            "[ranker-trainer] shared inference ranker ready "
            f"name={self.ranker_inference_actor_name} status={status}",
            flush=True,
        )
        self._maybe_sync_shared_inference_ranker(force=True)

    def _ranker_local_step(self) -> int:
        if getattr(self, "ranker_wg", None) is None:
            return 0
        worker = getattr(self.ranker_wg, "worker", None)
        return int(getattr(worker, "step", 0) or 0)

    def _maybe_sync_shared_inference_ranker(self, *, force: bool = False) -> dict:
        if getattr(self, "ranker_inference_actor", None) is None:
            return {}
        sync_interval = self._ranker_inference_sync_interval()
        local_step = self._ranker_local_step()
        if sync_interval <= 0:
            return {"ranker/inference_sync_enabled": 0}
        if not force:
            if local_step <= 0:
                return {"ranker/inference_sync_enabled": 1}
            if local_step == self.ranker_inference_last_sync_step:
                return {"ranker/inference_sync_enabled": 1}
            if local_step % sync_interval != 0:
                return {"ranker/inference_sync_enabled": 1}

        with getattr(self, "ranker_lock", contextlib.nullcontext()):
            state_dict = self.ranker_wg.export_encoder_state_cpu()
        sync_metrics = ray.get(
            self.ranker_inference_actor.load_encoder_state_cpu.remote(
                state_dict,
                step=local_step,
            )
        )
        self.ranker_inference_last_sync_step = local_step
        sync_metrics.update(
            {
                "ranker/inference_sync_enabled": 1,
                "ranker/inference_sync_performed": 1,
                "ranker/inference_sync_interval": sync_interval,
            }
        )
        return sync_metrics

    def _async_ranker_training_enabled(self) -> bool:
        return bool(_cfg_require(self.config, "ranker_training.async_ranker_training.enable")) and (
            _cfg_require(self.config, "ranker_training.signal_source") == "async_ranker_training"
        )

    def _background_ranker_thread_enabled(self) -> bool:
        if not self._async_ranker_training_enabled():
            return False
        return bool(_cfg_require(self.config, "ranker_training.async_ranker_training.background_ranker_thread"))

    def _async_ranker_updates_per_global_step(self) -> int:
        value = int(_cfg_require(self.config, "ranker_training.async_ranker_training.ranker_updates_per_global_step"))
        if value <= 0:
            raise ValueError("ranker_training.async_ranker_training.ranker_updates_per_global_step must be >= 1")
        return value

    def _async_ranker_training_log_dir(self) -> str:
        configured = _cfg_require_present(self.config, "ranker_training.async_ranker_training.logging.log_dir")
        if configured:
            return str(configured)
        rollout_dir = str(self.config.trainer.get("rollout_data_dir", ""))
        if rollout_dir:
            run_dir = os.path.dirname(rollout_dir)
        else:
            run_dir = str(self.config.trainer.default_local_dir)
        return os.path.join(run_dir, "async_ranker_training")

    def _save_checkpoint(self):
        super()._save_checkpoint()
        if getattr(self, "ranker_wg", None) is None:
            return
        ranker_path = os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{self.global_steps}",
            "ranker",
        )
        if getattr(self, "ranker_async_trainer", None) is not None:
            self.ranker_async_trainer.save_checkpoint(ranker_path)
        else:
            with getattr(self, "ranker_lock", contextlib.nullcontext()):
                self.ranker_wg.save_checkpoint(ranker_path)

    def _enrich_tool_calls_with_ranker(self, main_batch):
        """Add dense ranker top50 traces before sample construction."""
        if getattr(self, "ranker_wg", None) is None:
            return
        non_tensor = getattr(main_batch, "non_tensor_batch", {}) or {}
        if "tool_call_details" not in non_tensor:
            return

        details_list = _as_python_list(non_tensor.get("tool_call_details"))
        enriched_tool_calls = 0
        ranked_docs_count = 0
        for tool_details in details_list:
            for detail in _as_python_list(tool_details):
                if not isinstance(detail, dict):
                    continue
                recall_docs = (
                    detail.get("recall_top50_docs")
                    or detail.get("top_50_documents")
                    or detail.get("top_n_documents")
                    or []
                )
                recall_docs = _as_python_list(recall_docs)[: int(_cfg_require(self.config, "recall_retriever.top_k"))]
                if not recall_docs:
                    continue
                sub_query = str(detail.get("sub_query") or "")
                if not sub_query:
                    continue
                if "recall_top50_docs" not in detail:
                    detail["recall_top50_docs"] = recall_docs
                with getattr(self, "ranker_lock", contextlib.nullcontext()):
                    rank_top50 = self.ranker_wg.worker.rank_topk(
                        query=sub_query,
                        docs=recall_docs,
                        top_k=len(recall_docs),
                        max_query_length=int(_cfg_require(self.config, "ranker.max_query_length")),
                        max_doc_length=int(_cfg_require(self.config, "ranker.max_doc_length")),
                    )
                if not rank_top50:
                    continue
                detail["rank_top50_docs"] = rank_top50
                detail["rank_top5_docs"] = rank_top50[: int(_cfg_require(self.config, "ranker.top_k"))]
                detail["ranked_passages"] = rank_top50
                enriched_tool_calls += 1
                ranked_docs_count += len(rank_top50)

        main_batch.meta_info["ranker_trace"] = {
            "enriched_tool_calls": enriched_tool_calls,
            "ranked_docs": ranked_docs_count,
        }

    def _submit_async_ranker_training_requests(self, main_batch) -> dict:
        if getattr(self, "async_ranker_labeler", None) is None:
            return {}
        self.async_ranker_labeler.update_global_step(self.global_steps)
        prompt_version = ""
        stages = _cfg_require(self.config, "ranker_training.async_ranker_training.stages") or []
        for stage in stages:
            if isinstance(stage, dict) and stage.get("type") == "llm_as_judge":
                prompt_version = str((stage.get("prompt") or {}).get("version") or "")
                break
            if hasattr(stage, "get") and stage.get("type") == "llm_as_judge":
                prompt_version = str((stage.get("prompt") or {}).get("version") or "")
                break
        fresh_trajectories = build_fresh_trajectories_from_dataproto(main_batch, self.global_steps)
        selected_contexts = self.async_trajectory_selector.select(fresh_trajectories)
        candidate_tool_calls = sum(len(trajectory.get("tool_calls", [])) for trajectory in fresh_trajectories)
        requests, build_metrics = build_requests_from_contexts(
            selected_contexts,
            global_step=self.global_steps,
            max_sub_query=int(_cfg_require(self.config, "ranker_training.async_ranker_training.max_sub_query")),
            prompt_version=prompt_version,
            sub_query_selection_policy=str(
                _cfg_require(self.config, "ranker_training.async_ranker_training.sub_query_selection_policy")
            ),
            selection_seed=int(_cfg_require(self.config, "ranker_training.async_ranker_training.selection_seed")),
            label_policy=str(_cfg_require(self.config, "ranker_training.async_ranker_training.label_policy")),
        )
        accepted = self.async_ranker_labeler.submit(requests)
        return {
            "async_ranker_training/candidate_tool_calls": candidate_tool_calls,
            "async_ranker_training/selector_contexts": build_metrics.get("candidate_tool_calls", 0),
            "async_ranker_training/selected_tool_calls": build_metrics.get("selected_tool_calls", 0),
            "async_ranker_training/invalid_requests": build_metrics.get("invalid_requests", 0),
            "async_ranker_training/built_requests": len(requests),
            "async_ranker_training/accepted_requests": accepted,
        }

    def _async_ranker_metrics(self) -> dict:
        if getattr(self, "ranker_async_trainer", None) is None:
            return {}
        return self.ranker_async_trainer.get_metrics()

    def _run_async_ranker_update_once(self) -> dict:
        if getattr(self, "ranker_async_trainer", None) is None:
            return {}
        updated = self.ranker_async_trainer.try_train_once(wait=False, timeout=0.0)
        metrics = self.ranker_async_trainer.get_metrics()
        metrics["ranker/async_background_thread"] = 0
        metrics["ranker/async_attempted_update"] = 1
        metrics["ranker/async_updated_this_step"] = 1 if updated else 0
        if updated:
            metrics.update(self._maybe_sync_shared_inference_ranker())
        return metrics

    def _run_async_ranker_updates_for_step(self) -> dict:
        if getattr(self, "ranker_async_trainer", None) is None:
            return {}
        max_updates = self._async_ranker_updates_per_global_step()
        metrics = {}
        attempted = 0
        updated_count = 0
        for _ in range(max_updates):
            attempted += 1
            updated = self.ranker_async_trainer.try_train_once(wait=False, timeout=0.0)
            step_metrics = self.ranker_async_trainer.get_metrics()
            if updated:
                updated_count += 1
                step_metrics.update(self._maybe_sync_shared_inference_ranker())
                metrics.update(step_metrics)
                continue
            metrics.update(step_metrics)
            break
        metrics["ranker/async_background_thread"] = 0
        metrics["ranker/async_attempted_update"] = attempted
        metrics["ranker/async_updated_this_step"] = 1 if updated_count > 0 else 0
        metrics["ranker/async_updates_this_step"] = updated_count
        metrics["ranker/async_updates_per_global_step"] = max_updates
        return metrics

    def _collect_background_ranker_metrics(self) -> dict:
        if getattr(self, "ranker_async_trainer", None) is None:
            return {}
        metrics = self.ranker_async_trainer.get_metrics()
        updates = int(metrics.get("ranker/async_updates", 0) or 0)
        updates_since_last_log = max(0, updates - int(self.ranker_async_last_logged_updates))
        self.ranker_async_last_logged_updates = updates
        metrics["ranker/async_background_thread"] = 1
        metrics["ranker/async_attempted_update"] = 0
        metrics["ranker/async_updated_this_step"] = 1 if updates_since_last_log > 0 else 0
        metrics["ranker/async_updates_since_last_log"] = updates_since_last_log
        return metrics

    def _close_async_components(self):
        if getattr(self, "ranker_async_trainer", None) is not None:
            self.ranker_async_trainer.stop()
        if getattr(self, "async_ranker_labeler", None) is not None:
            self.async_ranker_labeler.close()

    def fit(self):
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        current_epoch = self.global_steps // len(self.train_dataloader)

        if (self.val_reward_fn or self.use_reward_loop) and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            from verl.utils.rollout_skip import RolloutSkip

            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                ranker_train_enabled = self._ranker_train_enabled()
                metrics["ranker/enabled"] = 1 if ranker_train_enabled else 0

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        assert self.async_rollout_mode, "Only async rollout mode is supported in CoAgentic ranker trainer"
                        main_batch = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        if "timing" in main_batch.meta_info:
                            timing_raw.update({f"main_{k}": v for k, v in main_batch.meta_info["timing"].items()})
                            main_batch.meta_info.pop("timing", None)
                        if "aggregated_metrics" in main_batch.meta_info:
                            metrics.update({f"main_{k}": v for k, v in main_batch.meta_info.pop("aggregated_metrics").items()})

                        if ranker_train_enabled:
                            self._enrich_tool_calls_with_ranker(main_batch)
                            ranker_trace = main_batch.meta_info.pop("ranker_trace", {})
                            if ranker_trace:
                                metrics.update(
                                    {
                                        "ranker/trace_enriched_tool_calls": ranker_trace.get("enriched_tool_calls", 0),
                                        "ranker/trace_ranked_docs": ranker_trace.get("ranked_docs", 0),
                                    }
                                )
                            if self._async_ranker_training_enabled():
                                metrics.update(self._submit_async_ranker_training_requests(main_batch))

                    if "response_mask" not in main_batch.batch.keys():
                        main_batch.batch["response_mask"] = compute_response_mask(main_batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(main_batch, metrics=metrics)

                    main_ref_wg = getattr(self, "ref_policy_wg", None) if self.use_reference_policy else None
                    main_futures = process_main_agent_ppo_step.remote(
                        batch=main_batch,
                        actor_rollout_wg=self.actor_rollout_wg,
                        ref_policy_wg=main_ref_wg,
                        critic_wg=self.critic_wg if self.use_critic else None,
                        tokenizer=self.tokenizer,
                        config=self.config,
                        global_steps=self.global_steps,
                        use_reference_policy=self.use_reference_policy,
                        use_critic=self.use_critic,
                        kl_ctrl_in_reward=self.kl_ctrl_in_reward,
                        ref_in_actor=self.ref_in_actor,
                    )

                    if ranker_train_enabled and not self._async_ranker_training_enabled():
                        with marked_timer("ranker_contrastive_total", timing_raw, color="blue"):
                            fresh_trajectories = build_fresh_trajectories_from_dataproto(main_batch, self.global_steps)
                            steps_per_global = int(_cfg_require(self.config, "trainer.ranker_steps_per_global_step"))
                            for ranker_step_idx in range(steps_per_global):
                                ranker_metrics = process_ranker_contrastive_step(
                                    fresh_trajectories=fresh_trajectories,
                                    ranker_wg=self.ranker_wg,
                                    replay_buffer=self.ranker_replay_buffer,
                                    selector=self.ranker_selector,
                                    signal_builder=self.ranker_signal_builder,
                                    sample_builder=self.ranker_sample_builder,
                                    collator=self.ranker_collator,
                                    config=self.config,
                                    global_steps=self.global_steps,
                                    ranker_step_idx=ranker_step_idx,
                                    construction_logger=self.ranker_construction_logger,
                                )
                                self.ranker_update_step += 1
                                metrics.update({f"{k}_step_{ranker_step_idx}": v for k, v in ranker_metrics.items() if k.startswith("ranker/loss")})
                                metrics.update(ranker_metrics)
                                metrics.update(self._maybe_sync_shared_inference_ranker())
                            metrics["ranker/steps_per_global_step"] = steps_per_global
                    elif ranker_train_enabled:
                        metrics["ranker/async_mode"] = 1
                        if self._background_ranker_thread_enabled():
                            with marked_timer("ranker_async_metrics", timing_raw, color="blue"):
                                metrics.update(self._collect_background_ranker_metrics())
                        else:
                            with marked_timer("ranker_async_update_for_step", timing_raw, color="blue"):
                                metrics.update(self._run_async_ranker_updates_for_step())

                    main_results = ray.get(main_futures)
                    main_batch, main_metrics, main_timing_raw, main_reward_extra_infos_dict = main_results

                    metrics.update(main_metrics)
                    timing_raw.update(main_timing_raw)

                    main_rollout_stats = self._compute_rollout_stats(main_batch, main_reward_extra_infos_dict, prefix="main_agent")
                    metrics.update(main_rollout_stats)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if self._should_dump_rollout_data(is_last_step=is_last_step):
                        rollout_dump_dir = os.path.join(rollout_data_dir, "main_agent")
                        rollout_dump_batch = self._limit_rollout_dump_batch(main_batch)
                        if len(rollout_dump_batch) > 0:
                            self._log_rollout_data(
                                rollout_dump_batch,
                                self._limit_rollout_dump_reward_infos(
                                    main_reward_extra_infos_dict,
                                    len(rollout_dump_batch),
                                ),
                                timing_raw,
                                rollout_dump_dir,
                            )
                            self._prune_rollout_dump_files(rollout_dump_dir)

                if (
                    (self.val_reward_fn or self.use_reward_loop)
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(compute_data_metrics(batch=main_batch, use_critic=self.use_critic, agent_name="main_agent"))
                metrics.update(compute_timing_metrics(batch=main_batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=main_batch, timing_raw=timing_raw, n_gpus=n_gpus))
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    self._close_async_components()
                    return
