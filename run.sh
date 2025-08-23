# CUDA_VISIBLE_DEVICES=0 DEEPCOMPRESSOR_TOKEN_NLL_BATCH_SIZE=1 uv run python -m deepcompressor.app.llm.ptq examples/llm/configs/awq.yaml \
#     --model-name llama-3-8b --model-path /dev/shm/Meta-Llama-3-8B \
#     --delta-single-layer true

CUDA_VISIBLE_DEVICES=1 DEEPCOMPRESSOR_TOKEN_NLL_BATCH_SIZE=1 uv run python -m deepcompressor.app.llm.ptq examples/llm/configs/awq.yaml \
    --model-name llama-2-7b --model-path /dev/shm/Llama-2-7b-hf \
    --delta-single-layer true