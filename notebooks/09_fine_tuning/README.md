# Chapter 9 — Fine-Tuning RAG Data with Human Feedback

Fine-tuning an OpenAI model on RAG-derived data and human feedback, and
comparing it against pure retrieval.

Source notebook:
- [Fine_tuning_OpenAI_GPT-4o-mini.ipynb](https://github.com/Denis2054/RAG-Driven-Generative-AI/blob/main/Chapter09/Fine_tuning_OpenAI_GPT-4o-mini.ipynb)

Tools: OpenAI fine-tuning API.

This is the one chapter that touches model training directly (via OpenAI's
hosted fine-tuning, not local `deepspeed`/`bitsandbytes` training — those are
installed for local Hugging Face model work if you extend the project that
way).
