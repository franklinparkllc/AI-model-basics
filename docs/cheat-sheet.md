# The AI Architecture Cheat Sheet

## Part 3: Quick Reference Guide

### Core Concepts

| Term | Definition | Simple Analogy |
|------|------------|----------------|
| **Token** | A fragment of a word (the model's basic unit). | Reusable puzzle pieces. |
| **Embedding** | A mathematical "scouting report" for a token. | A flavor profile (sweet, salty, etc.). |
| **Latent Space** | A multi-dimensional map where tokens live. | A neighborhood of meaning. |
| **Attention** | A mechanism to weight the importance of words. | A highlighter pen for context. |
| **Transformer** | The standard architecture for modern LLMs. | A skyscraper of processing floors. |
| **Pre-training** | Learning from trillions of tokens of raw data. | A general education (reading everything). |
| **Post-training** | Fine-tuning a model to be helpful and safe. | Professional etiquette training. |
| **Inference** | The process of the model generating a response. | Querying a frozen digital brain. |
| **RAG** | Fetching external data to inform a response. | An open-book exam. |
| **Temperature** | A setting that controls randomness/creativity. | A dice roll for the next word. |

---

## Expanded Definitions

### Tokenization & Text Processing

**Token**
- **Technical Definition:** The smallest unit of text that a model processes. Can be a word, subword, or character.
- **Example:** "Ingenious" → ["In", "gen", "ious"] (3 tokens)
- **Why It Matters:** Token count determines API costs and context limits. Different models use different tokenizers.
- **Common Tokenizers:** BPE (Byte Pair Encoding), SentencePiece, WordPiece

**Embedding**
- **Technical Definition:** A dense vector representation of a token in high-dimensional space (typically 512-4096 dimensions).
- **Properties:** Similar words have similar embeddings (measured by cosine similarity).
- **Training:** Learned during pre-training through self-supervised learning.
- **Use Cases:** Semantic search, similarity matching, feature extraction.

**Latent Space**
- **Technical Definition:** The high-dimensional vector space (typically 512-4096 dimensions) where embeddings exist.
- **Properties:** 
  - Geometric relationships encode semantic relationships
  - Linear operations can represent semantic operations (e.g., king - man + woman ≈ queen)
- **Visualization:** Often reduced to 2D/3D using techniques like t-SNE or UMAP for visualization.

---

### Architecture Components

**Attention Mechanism**
- **Purpose:** Allows the model to focus on relevant parts of the input when processing each token.
- **Types:**
  - **Self-Attention:** Tokens attend to other tokens in the same sequence
  - **Cross-Attention:** Tokens attend to tokens in a different sequence (e.g., encoder-decoder)
  - **Multi-Head Attention:** Multiple attention mechanisms run in parallel, each focusing on different aspects
- **QKV (Query, Key, Value):**
  - **Query (Q):** "What am I looking for?"
  - **Key (K):** "What do I represent?"
  - **Value (V):** "What information do I carry?"
- **Formula:** Attention(Q, K, V) = softmax(QK^T / √d_k) V

**Transformer Architecture**
- **Components:**
  - **Encoder:** Processes input (used in BERT, T5)
  - **Decoder:** Generates output (used in GPT, PaLM)
  - **Encoder-Decoder:** Both (used in T5, BART)
- **Layer Structure:**
  1. **Multi-Head Self-Attention:** Token-to-token communication
  2. **Feed-Forward Network:** Per-token processing
  3. **Layer Normalization:** Stabilizes training
  4. **Residual Connections:** Helps with gradient flow
- **Stacking:** Models typically have 12-96+ layers, with each layer adding depth and nuance.

---

### Training Phases

**Pre-training**
- **Objective:** Learn general language patterns by predicting the next token (or masked tokens).
- **Data:** Massive text corpora (Common Crawl, books, code, etc.) - often trillions of tokens.
- **Process:** Self-supervised learning - no human labels needed.
- **Outcome:** Base model with broad but unfocused knowledge.
- **Examples:** GPT-3, LLaMA base models, BERT base.

**Post-training (Fine-tuning)**
- **Objective:** Adapt the base model for specific tasks or behaviors.
- **Methods:**
  - **Supervised Fine-Tuning (SFT):** Train on labeled examples
  - **Reinforcement Learning from Human Feedback (RLHF):** Optimize for human preferences
  - **Direct Preference Optimization (DPO):** More efficient alternative to RLHF
- **Use Cases:** 
  - Making models helpful, harmless, and honest
  - Adapting to specific domains (legal, medical, etc.)
  - Teaching specific formats or styles
- **Examples:** ChatGPT (RLHF), Claude (Constitutional AI), instruction-tuned models.

---

### Inference & Generation

**Inference**
- **Definition:** The process of using a trained model to generate predictions/responses.
- **Key Point:** Model weights are frozen - no learning happens during inference.
- **Process:**
  1. Tokenize input prompt
  2. Convert tokens to embeddings
  3. Pass through transformer layers
  4. Generate probability distribution over vocabulary
  5. Sample next token
  6. Repeat until completion

**Temperature**
- **Range:** Typically 0.0 to 2.0
- **Low Temperature (0.0-0.3):**
  - More deterministic, predictable outputs
  - Good for factual tasks, code generation
  - Picks highest probability tokens
- **High Temperature (0.7-2.0):**
  - More creative, diverse outputs
  - Good for creative writing, brainstorming
  - Explores lower-probability options
- **Default:** Usually 0.7-1.0 for balanced creativity/consistency

**Top-p (Nucleus Sampling)**
- **Definition:** Alternative to temperature that considers only tokens whose cumulative probability exceeds p.
- **Example:** Top-p = 0.9 means consider tokens until their cumulative probability reaches 90%.
- **Use:** Often combined with temperature for better control.

**Top-k Sampling**
- **Definition:** Only consider the k most likely tokens.
- **Example:** Top-k = 50 means only sample from the 50 highest-probability tokens.
- **Use:** Prevents very unlikely tokens from being selected.

---

### Advanced Techniques

**RAG (Retrieval-Augmented Generation)**
- **Components:**
  1. **Retrieval:** Search relevant documents/knowledge base
  2. **Augmentation:** Inject retrieved context into prompt
  3. **Generation:** Model generates answer using context
- **Benefits:**
  - Access to up-to-date information
  - Traceability (can cite sources)
  - Domain-specific knowledge without retraining
- **Architecture:**
  - Vector database (e.g., Pinecone, Weaviate)
  - Embedding model for document indexing
  - LLM for generation
- **Use Cases:** Knowledge bases, documentation Q&A, research assistants

**Fine-Tuning vs. RAG**
- **Fine-Tuning:** Model learns new information/patterns (permanent change)
- **RAG:** Model accesses external information (temporary context)
- **Choose Fine-Tuning When:** Need style change, domain-specific reasoning, consistent formatting
- **Choose RAG When:** Facts change frequently, need citations, multiple knowledge sources

**Reasoning Models**
- **Definition:** Models that use "inference-time compute" to think before responding.
- **Techniques:**
  - **Chain-of-Thought (CoT):** Step-by-step reasoning
  - **Tree-of-Thoughts:** Explore multiple reasoning paths
  - **Self-Consistency:** Generate multiple answers and vote
- **Examples:** GPT-4 with reasoning, Claude with extended thinking, o1 models
- **Trade-off:** Slower inference but better accuracy on complex problems

---

### Model Types & Sizes

**Model Size Categories**

| Size | Parameters | Use Cases | Hardware |
|------|-----------|-----------|----------|
| **Small** | 1B - 7B | Edge devices, specific tasks, fast inference | Consumer GPUs, CPUs |
| **Medium** | 13B - 70B | General purpose, business applications | Server GPUs (A100, H100) |
| **Large** | 100B+ | Maximum capability, research | Multi-GPU clusters, TPUs |

**Popular Models by Size:**
- **Small:** GPT-2 (1.5B), LLaMA-2-7B, Mistral-7B
- **Medium:** LLaMA-2-70B, PaLM-2, GPT-3.5
- **Large:** GPT-4, Claude-3 Opus, PaLM-2 (540B)

---

### Key Metrics & Evaluation

**Perplexity**
- **Definition:** Measure of how "surprised" the model is by the test data.
- **Lower = Better:** Lower perplexity means better language modeling.
- **Use:** Compare different models or training configurations.

**BLEU Score**
- **Definition:** Measures similarity between generated and reference text (for translation/summarization).
- **Range:** 0-1 (or 0-100), higher is better.
- **Limitations:** Doesn't capture semantic meaning well.

**ROUGE Score**
- **Definition:** Measures overlap of n-grams (for summarization).
- **Variants:** ROUGE-1 (unigrams), ROUGE-2 (bigrams), ROUGE-L (longest common subsequence).

**Human Evaluation**
- **Metrics:** Helpfulness, harmlessness, honesty, factual accuracy.
- **Methods:** Pairwise comparisons, rating scales, A/B testing.

---

### Common Acronyms

| Acronym | Full Form | Description |
|---------|-----------|-------------|
| **LLM** | Large Language Model | General term for large AI language models |
| **GPT** | Generative Pre-trained Transformer | OpenAI's model series |
| **BERT** | Bidirectional Encoder Representations from Transformers | Google's encoder model |
| **RLHF** | Reinforcement Learning from Human Feedback | Training method for alignment |
| **DPO** | Direct Preference Optimization | Alternative to RLHF |
| **RAG** | Retrieval-Augmented Generation | Technique for adding external knowledge |
| **API** | Application Programming Interface | How to access models programmatically |
| **GPU** | Graphics Processing Unit | Hardware for training/inference |
| **TPU** | Tensor Processing Unit | Google's specialized AI chip |
| **PII** | Personally Identifiable Information | Data that identifies individuals |
| **CoT** | Chain-of-Thought | Reasoning technique |
| **SFT** | Supervised Fine-Tuning | Training on labeled examples |

---

### Quick Decision Tree

**Choosing a Model Size:**
```
Do you need maximum capability?
├─ Yes → Large model (100B+)
└─ No → Do you have GPU infrastructure?
    ├─ Yes → Medium model (13-70B)
    └─ No → Small model (1-7B) or API
```

**Choosing RAG vs. Fine-Tuning:**
```
Does your data change frequently?
├─ Yes → RAG
└─ No → Do you need style/format adaptation?
    ├─ Yes → Fine-Tuning
    └─ No → Base model may suffice
```

**Choosing Temperature:**
```
What's your use case?
├─ Factual/Coding → Low (0.0-0.3)
├─ Balanced → Medium (0.7-1.0)
└─ Creative → High (1.0-2.0)
```

---

### Best Practices Checklist

**For Prompt Engineering:**
- [ ] Be specific about the task
- [ ] Provide relevant context
- [ ] Include examples when helpful
- [ ] Specify output format
- [ ] Set appropriate constraints
- [ ] Iterate and refine

**For Production Deployment:**
- [ ] Implement error handling
- [ ] Set rate limits
- [ ] Monitor costs and usage
- [ ] Add logging and analytics
- [ ] Implement caching where appropriate
- [ ] Set up monitoring and alerts
- [ ] Plan for scaling
- [ ] Consider data privacy and security

**For Model Selection:**
- [ ] Match model size to task complexity
- [ ] Consider inference latency requirements
- [ ] Evaluate cost vs. capability trade-offs
- [ ] Test on your specific use case
- [ ] Consider fine-tuning vs. prompt engineering
- [ ] Plan for model updates and maintenance

---

### Common Pitfalls to Avoid

1. **Assuming models know truth:** They predict patterns, not facts
2. **Ignoring context limits:** Models have maximum token limits
3. **Using wrong temperature:** Too high for factual tasks, too low for creative
4. **Not using RAG when needed:** For frequently changing information
5. **Overlooking costs:** API costs can scale quickly
6. **Ignoring privacy:** Understand data retention policies
7. **Not testing thoroughly:** Models can fail in unexpected ways
8. **Forgetting about latency:** Large models are slower
9. **Not planning for updates:** Model capabilities improve rapidly
10. **Underestimating prompt engineering:** Small prompt changes = big output differences

---

*Last Updated: February 2025*
