参考/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/AgenticIterRag/main_co_search_ppo.py的架构和风格进行实现。

此脚本为AgenticIterRag项目的核心入口训练脚本；此项目的目标是进行agent rag的交替训练实验；具体说，是agent llm和reranker llm交替训练；

1. 解析config

2. 根据核心config参数，编排训练流程：
    核心config参数1：交替训练轮数: N_ITER; N_ITER表示交替训练的总轮数；每轮包含一次agent llm训练和一次reranker llm训练；
    2.1. 根据N_ITER、epoch数、max_steps、batch size等参数，计算agent llm训练的总step数；同样计算reranker llm训练的总step数；
        2.1.1 
    