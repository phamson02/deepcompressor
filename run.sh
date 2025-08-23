poetry run python -m deepcompressor.app.llm.ptq examples/llm/configs/awq.yaml \
    --model-name llama-2-7b --model-path /dev/shm/Llama-2-7b-hf \
    --delta-single-layer true