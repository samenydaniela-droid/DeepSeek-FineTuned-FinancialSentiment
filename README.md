# DeepSeek Fine-Tuned for Financial Sentiment Analysis

LoRA fine-tuning of **DeepSeek-R1-Distill-Qwen-1.5B** on the **Financial PhraseBank** dataset, with 4-bit QLoRA, on a single Colab T4 GPU.

> Aivancity School for Technology, Business and Society — 2025–2026  
> Authors: Daniela Sameny, Alexandre Georges Lissoko

## Results

Evaluated on a 300-sentence test subset of Financial PhraseBank (`sentences_allagree`):

| Method                       | Accuracy   | F1 (weighted) |
|------------------------------|------------|---------------|
| Zero-shot (base model)       | 54.00%     | 45.32%        |
| **Fine-tuned (LoRA, ours)**  | **84.33%** | **84.17%**    |
| **Δ Improvement**            | **+30.33 pts** | **+38.85 pts** |

Relative improvement: **+56.2% accuracy** / **+85.7% F1**.

## Method

- Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Adapter: LoRA (rank=8, alpha=16, dropout=0.05) on attention projections (q, k, v, o)
- Quantization: 4-bit NF4 (QLoRA)
- Dataset: Financial PhraseBank (Malo et al. 2014), `sentences_allagree` subset
- Training: 1 epoch, batch size 16 effective, learning rate 2e-4, cosine schedule, bf16
- Hardware: NVIDIA T4 (16 GB), Google Colab
- Random seed: 42

## Repository contents

- `DeepSeek_FineTuning.ipynb` — full Colab pipeline
- `deepseek-finbert-lora/` — trained LoRA adapter (4.2 MB)
- `finetuning_results.csv` — evaluation metrics
- `DeepSeek_FineTuning.pdf` — paper

## How to use the fine-tuned adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)

base = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, "./deepseek-finbert-lora")
tokenizer = AutoTokenizer.from_pretrained("./deepseek-finbert-lora")
```

## License

MIT for the code. Financial PhraseBank dataset is licensed under CC BY-NC-SA 3.0.
