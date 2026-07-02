export XXX_<env_param>="" # 环境配置字段写在这里，如task_name、group_name之类的
.....
bash "${ROOT}/scripts/coagenticRetriever_v2/01_train_launcher.sh"
    --xx_CONFIG=xxx/xxx.yaml \ # 支持使用新的yanl
    --actor_rollout_ref.rollout.multi_turn.xxxx=xx \ # 支持对具体yaml文件内的配置项进行改动