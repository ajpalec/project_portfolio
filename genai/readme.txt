The `llm` directory contains subdirectories and files for:
* performing retrieval augmented generation using a Postgres-based vector database (PGVector) without and with LangChain API
* performing parameter-efficient fine-tuning (PEFT) through quantized Low Rank Adaptation (QLoRA) of a Mistral 7-billion parameter large language model
    * fine-tuned on dolly-15k dataset
* Gradio interactive chatbot application relying on the model