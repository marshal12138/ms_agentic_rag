
corpus_file=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/retrieval/wiki-18/wiki-18.jsonl # jsonl
save_dir=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/retrieval/wiki-18/bm25_index
retriever_name=bm25 # this is for indexing naming
retriever_model=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PATH=/data04/envs/ms/ms_cosearch_official/bin:$PATH \
/data04/envs/ms/ms_cosearch_official/bin/python index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding
