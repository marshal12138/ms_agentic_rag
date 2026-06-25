from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import asyncio
import os
from langchain_core.runnables import chain
import uvicorn
import torch
from modelscope import AutoModel, AutoTokenizer, AutoModelForCausalLM
import threading
import time

app = FastAPI(title="检索服务")

# 配置
# 指向独立的 dense retriever 实例（默认 8040），与训练直连的 8030 物理隔离，
# 避免 8033 回调打到 8030 造成线程池递归占用 / 队列雪崩。
RETRIEVER_URL = os.getenv("HEAVY_UPSTREAM_RETRIEVER_URL", "http://localhost:8040/retrieve")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 50  # 最终返回50个

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int

# ============ Qwen3-Reranker 初始化 ============
class Qwen3Reranker:
    def __init__(self, model_name="/data01/ms_wksp/agent_up_to_date/models/reranker/Qwen3-Reranker-8B", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {self.device}")
        
        # 加载模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        ).eval()
        
        # 如果有GPU，可以启用flash attention加速
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name, 
        #     torch_dtype=torch.float16, 
        #     attn_implementation="flash_attention_2"
        # ).cuda().eval()
        
        # 获取特殊token的ID
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192
        
        # 构建prompt模板
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        
        # 任务指令
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'
        
        print(f"Qwen3-Reranker-8B 加载完成，设备: {self.device}")
    
    def format_instruction(self, query: str, doc: Dict[str, Any]) -> str:
        """格式化指令"""
        doc_text = doc.get("contents", "")
        instruction = self.task
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=instruction, query=query, doc=doc_text
        )
        return output
    
    def process_inputs(self, pairs: List[str]):
        """处理输入"""
        # 计算实际最大长度
        max_len = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        
        # Tokenize
        inputs = self.tokenizer(
            pairs, 
            padding=False, 
            truncation='longest_first',
            return_attention_mask=False, 
            max_length=max_len
        )
        
        # 添加prefix和suffix
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        
        # Padding
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        
        # 移动到设备
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        
        return inputs
    
    @torch.no_grad()
    def compute_logits(self, inputs):
        """计算评分"""
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        del true_vector, false_vector, batch_scores
        return scores
    
    def rerank(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对文档进行重排序
        """
        if not documents:
            return documents
        
        # 限制重排序数量，避免GPU内存溢出
        rerank_docs = documents[:100]
        
        try:
            # 构建pairs
            pairs = [self.format_instruction(query, doc) for doc in rerank_docs]
            
            # 处理输入
            inputs = self.process_inputs(pairs)
            
            # 计算分数
            scores = self.compute_logits(inputs)
            del inputs
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # 将分数添加到文档中
            for doc, score in zip(rerank_docs, scores):
                doc["relevance_score"] = score
            
            # 按分数降序排序
            reranked_docs = sorted(rerank_docs, key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            return reranked_docs
            
        except Exception as e:
            print(f"重排序失败: {e}")
            # 如果重排序失败，返回原始顺序
            return documents

# 全局重排序器实例
reranker = None

# ============ 检索函数 ============
@chain
def ensemble_retriever(query: str) -> List[Dict[str, Any]]:
    """
    使用langchain的chain装饰器，执行混合检索
    """
    import requests
    
    payload1 = {
        "queries": [query],
        "dense_weight": 1.0,
        "bm25_weight": 0.0,
        "heavy_weight": 0.0,
        "topk": 100
    }
    payload2 = {
        "queries": [query],
        "dense_weight": 0.0,
        "bm25_weight": 1.0,
        "heavy_weight": 0.0,
        "topk": 100
    }
    
    try:
        response1 = requests.post(RETRIEVER_URL, json=payload1, timeout=30)
        response2 = requests.post(RETRIEVER_URL, json=payload2, timeout=30)
        
        results1 = response1.json() if response1.status_code == 200 else {"result": [[]]}
        results2 = response2.json() if response2.status_code == 200 else {"result": [[]]}
        
        docs1 = results1.get("result", [[]])[0] if results1.get("result") else []
        docs2 = results2.get("result", [[]])[0] if results2.get("result") else []
        
        # 合并去重
        id_set = set()
        merged_docs = []
        
        for doc in docs1:
            doc_id = doc.get("id")
            if doc_id and doc_id not in id_set:
                merged_docs.append(doc)
                id_set.add(doc_id)
                
        for doc in docs2:
            doc_id = doc.get("id")
            if doc_id and doc_id not in id_set:
                merged_docs.append(doc)
                id_set.add(doc_id)
        
        return merged_docs
        
    except Exception as e:
        print(f"检索失败: {e}")
        return []

def rrf_fusion(results_list: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    """
    RRF (Reciprocal Rank Fusion) 融合多个排序列表
    """
    doc_scores = {}
    
    for rank_list in results_list:
        for rank, doc in enumerate(rank_list, start=1):
            doc_id = doc.get("id")
            if doc_id:
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {
                        "doc": doc,
                        "score": 0.0
                    }
                doc_scores[doc_id]["score"] += 1.0 / (k + rank)
    
    sorted_docs = sorted(
        doc_scores.values(), 
        key=lambda x: x["score"], 
        reverse=True
    )
    
    return [item["doc"] for item in sorted_docs]

# ============ FastAPI接口 ============
@app.on_event("startup")
async def startup_event():
    """服务启动时初始化重排序模型"""
    global reranker
    print("正在加载 Qwen3-Reranker-8B 模型...")
    start_time = time.time()
    reranker = Qwen3Reranker()
    print(f"模型加载完成，耗时: {time.time() - start_time:.2f}秒")

@app.post("/search")
async def heavy_recall(request: SearchRequest):
    """
    主要搜索接口：执行三次检索 + RRF融合 + 重排序
    """
    try:
        query = request.query
        top_k = min(request.top_k, 50)
        
        # 1. 第一次检索：使用ensemble_retriever（混合检索）
        docs1 = ensemble_retriever.invoke(query)
        
        # 2. 第二次检索：纯稠密检索
        payload_dense = {
            "queries": [query],
            "dense_weight": 1.0,
            "bm25_weight": 0.0,
            "heavy_weight": 0.0,
            "topk": 100
        }

        # 3. 第三次检索：纯BM25检索
        payload_bm25 = {
            "queries": [query],
            "dense_weight": 0.0,
            "bm25_weight": 1.0,
            "heavy_weight": 0.0,
            "topk": 100
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                client.post(RETRIEVER_URL, json=payload_dense),
                client.post(RETRIEVER_URL, json=payload_bm25)
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 提取文档列表
        docs2 = []
        docs3 = []
        
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception) or resp.status_code != 200:
                print(f"第{i+2}次检索失败")
                continue
            
            data = resp.json()
            docs = data.get("result", [[]])[0] if data.get("result") else []
            
            if i == 0:
                docs2 = docs
            else:
                docs3 = docs
        
        # 4. 合并三次召回结果（去重）
        id_set = set()
        all_docs = []
        
        for doc in docs1:
            doc_id = doc.get("id")
            if doc_id and doc_id not in id_set:
                all_docs.append(doc)
                id_set.add(doc_id)
        
        for doc in docs2:
            doc_id = doc.get("id")
            if doc_id and doc_id not in id_set:
                all_docs.append(doc)
                id_set.add(doc_id)
        
        for doc in docs3:
            doc_id = doc.get("id")
            if doc_id and doc_id not in id_set:
                all_docs.append(doc)
                id_set.add(doc_id)
        
        if not all_docs:
            return SearchResponse(results=[], total=0)
        
        # 5. RRF融合
        # 保留原始排序进行RRF
        docs1_original = docs1
        docs2_original = docs2
        docs3_original = docs3
        
        fused_docs = rrf_fusion([docs1_original, docs2_original, docs3_original])
        
        # 6. 重排序（使用本地Qwen3-Reranker）
        if reranker is not None:
            reranked_docs = reranker.rerank(query, fused_docs)
        else:
            reranked_docs = fused_docs
        
        # 7. 截取top_k，只返回id
        final_results = [{"id": doc.get("id")} for doc in reranked_docs[:top_k]]

        return SearchResponse(results=final_results, total=len(final_results))
        
    except Exception as e:
        print(f"搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8033,
        log_level="info"
    )