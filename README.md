# HedonicNeuron

This repository provides datasets and model checkpoints used for fine-tuning reranker models and probing large language models (LLMs) in our NeurIPS submission.

### 📚 Datasets

**Training Dataset**  
- Tevatron MSMARCO Passage Augmented  
  [HuggingFace Link](https://huggingface.co/datasets/Tevatron/msmarco-passage-aug)

**Evaluation Dataset**  
- TREC Deep Learning 2019  
  [Official Link](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html)

Fine-tuning follows the same procedure as outlined in the Tevatron MSMARCO implementation.

---

### 🧠 Finetuned Reranker Model Checkpoints

Below are the HuggingFace model IDs of the fine-tuned rerankers used in our experiments:

#### **Rank 8 Rerankers**
| Model | HuggingFace ID | Tokenizer |
|-------|----------------|-----------|
| Pythia | `AnonymousForReview2/finegrained_checkpoint_experiment_rankpythia_r8_mlp_only` | `EleutherAI/pythia-6.9b` |
| Mistral | `AnonymousForReview2/finegrained_checkpoint_experiment_rankmistral_r8_mlp_only` | `mistralai/Mistral-7B-v0.1` |
| LLaMA3 | `AnonymousForReview2/finegrained_checkpoint_experiment_llama3_r8_lora_mlp` | `meta-llama/Llama-3.1-8B` |

#### **CQTR Probes (Covered Query Term Ratio)**
| Model | HuggingFace ID | Tokenizer |
|-------|----------------|-----------|
| Pythia | `AnonymousForReview2/watereddown_reranker_pythia_cqtr_mlp_only` | `EleutherAI/pythia-6.9b` |
| Mistral | `AnonymousForReview2/watereddown_reranker_mistral_cqtr_mlp_only` | `mistralai/Mistral-7B-v0.1` |
| LLaMA3 | `AnonymousForReview2/watereddown_reranker_llama3_cqtr_mlp_only` | `meta-llama/Llama-3.1-8B` |

#### **Mean(tf/l) Probes**
| Model | HuggingFace ID | Tokenizer |
|-------|----------------|-----------|
| Pythia | `AnonymousForReview2/watereddown_reranker_pythia_tf_l_mlp_only` | `EleutherAI/pythia-6.9b` |
| Mistral | `AnonymousForReview2/watereddown_reranker_mistral_tf_l_mlp_only` | `mistralai/Mistral-7B-v0.1` |
| LLaMA3 | `AnonymousForReview2/watereddown_reranker_llama3_tf_l_mlp_only` | `meta-llama/Llama-3.1-8B` |

---

### ⚠️ Note

Accessing models based on **LLaMA3** requires appropriate permissions from Meta. Please ensure you have the necessary license agreements in place before attempting to use these checkpoints.

