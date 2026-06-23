"""Shared entry helpers for CoAgenticRetriever training."""

from __future__ import annotations

import warnings

import ray


def _allow_ray_without_faulthandler() -> None:
    """Let Ray continue if faulthandler cannot be enabled in this shell."""
    try:
        import ray._private.worker as ray_worker
    except Exception as exc:
        warnings.warn(f"Could not patch Ray faulthandler setup: {exc}", RuntimeWarning, stacklevel=2)
        return

    enable = ray_worker.faulthandler.enable
    if getattr(enable, "_coagentic_retriever_safe_enable", False):
        return

    def safe_enable(*args, **kwargs):
        try:
            return enable(*args, **kwargs)
        except OSError as exc:
            if getattr(exc, "errno", None) == 12:
                warnings.warn(
                    "Ray faulthandler setup failed with ENOMEM; continuing without faulthandler.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
            raise

    safe_enable._coagentic_retriever_safe_enable = True
    ray_worker.faulthandler.enable = safe_enable


class CoAgenticRetrieverTaskRunnerBase:
    """Worker/resource setup for a single main agent plus external ranker."""

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add the main agent actor-rollout worker."""
        from verl.single_controller.ray import RayWorkerGroup

        if config.actor_rollout_ref.rollout.mode == "sync":
            warnings.warn("spmd rollout mode is deprecated and will be removed in v0.6.2", stacklevel=2)

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
        else:
            raise NotImplementedError

        actor_rollout_cls = (
            AsyncActorRolloutRefWorker
            if config.actor_rollout_ref.rollout.mode == "async"
            else ActorRolloutRefWorker
        )

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        return actor_rollout_cls, RayWorkerGroup

    def add_critic_worker(self, config):
        """Add critic worker when the configured advantage estimator requires it."""
        if not config.critic.enable:
            return

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker
                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if not config.reward_model.enable:
            return

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl in ["auto", "enable"]:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
        elif use_legacy_worker_impl == "disable":
            from verl.workers.roles import RewardModelWorker
            print("Using new worker implementation")
        else:
            raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        self.mapping[Role.RewardModel] = "reward_pool" if config.reward_model.enable_resource_pool else "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker for KL loss or KL reward."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def init_resource_pool_mgr(self, config):
        """Use one pool for the main agent and optional reward-model resources."""
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")
            resource_pool_spec["reward_pool"] = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes

        self.mapping[Role.ActorRollout] = global_pool_id
        self.mapping[Role.Critic] = global_pool_id
        self.mapping[Role.RefPolicy] = global_pool_id

        return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)


DualAgentTaskRunner = CoAgenticRetrieverTaskRunnerBase
