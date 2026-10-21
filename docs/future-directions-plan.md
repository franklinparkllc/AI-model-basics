# Future Directions: Plan for Expanding "Looking Ahead"

**Status:** Planning notes for a future session. Nothing here is in the slides yet.
**Written:** 2026-10-21, right after the v26 accuracy pass on slides 1–22.
**Goal:** Replace the single "23. Looking Ahead" slide with a short **Frontier** section (about 6–8 slides). Then update `docs/cheat-sheet.yaml` to match both this section and the v26 corrections (checklist at the end).

> **Verify before publishing.** This field moves fast. Items marked ⚠️ come from secondary sources (blogs, news) or from memory and were *not* checked against a primary paper. Anything from after mid-2026 especially needs a fresh search. Check arXiv or the lab's own page before quoting numbers.

---

## Open question: what is "NCM"?

The original request mentioned "JEV or JEPA or NCM." **JEV** is almost certainly JEPA. **NCM** didn't match any known architecture in a web search. Candidates to ask the user about:

- **LCM**: Large Concept Models (Meta, Dec 2024). Predicts the next *sentence embedding* instead of the next token. A close cousin of JEPA. Covered below in Slide F5.
- **Neural Cellular Automata (NCA)**: self-organizing models; niche, probably not intended.
- **Nested Learning / Continuum Memory System (CMS)**: Google, 2025. Covered below in Slide F6.
- Something else (a product or company term).

Resolve this before building the section.

---

## Why a section, not a slide

Today's slide 23 lists five architectures in one breath (JEPA, Mamba/SSMs, MoE, RWKV/RetNet, hybrids). Each one answers a *different* limitation of the Transformer covered in slides 8–20. Giving each idea its own slide lets every frontier topic connect back to a specific earlier slide. That keeps the deck's teaching thread: **"here is the limitation you learned about → here is what researchers are trying."**

| Limitation taught earlier | Where it was taught | Frontier response |
|---|---|---|
| Predicts tokens, not ideas; no model of the world | Slides 3, 12, 13 | JEPA, world models, Large Concept Models |
| Attention cost grows with the square of context length | Slide 9 | SSMs / Mamba, linear attention, RWKV, hybrids |
| Every parameter runs for every token | Slides 8, 22 | Mixture of Experts |
| Generation is strictly one token at a time | Slides 8, 12 | Diffusion language models, multi-token prediction |
| Reasoning must be spelled out in tokens | Slide 20 | Latent reasoning (Coconut), small recursive models (HRM/TRM) |
| Weights are frozen after training | Slides 1, 10 | Test-time memory, continual learning (Titans, Nested Learning) |
| Tokenizers are blind to characters | Slide 3 | Byte-level models (Byte Latent Transformer) |
| Meaning lives in tangled directions (superposition) | Slide 4 | Mechanistic interpretability |

---

## Proposed slide breakdown

Numbering below assumes the Frontier section follows the Recap. An alternative is to put the Recap last; decide when building. Labels F1–F8 are placeholders.

### F1. Why Look Beyond the Transformer?

**Purpose:** A framing slide. Walk through the limitations table above in plain language.

- Transformers win on scale and simplicity, but have known costs: quadratic attention, token-by-token generation, frozen weights, and learning from text only.
- Two broad camps:
  1. **Make Transformers cheaper or faster.** MoE, SSM hybrids, diffusion decoding. These are already shipping.
  2. **Change what is predicted.** JEPA, world models, concept models. Mostly still research.
- Callout idea: *"Most frontier models in 2026 are still Transformers at heart. The frontier is mostly hybrids and new training methods, not wholesale replacement."*

### F2. JEPA and World Models

**Core idea:** Generative models predict every detail: the next token, or every pixel. JEPA (Joint Embedding Predictive Architecture) predicts the **embedding** (latent representation) of a missing part from the embedding of the visible part. It ignores unpredictable detail (the exact pattern of leaves on a tree) and focuses on what *can* be predicted (that the tree will still be there).

**Teaching hook:** This ties directly to Slide 4 (latent space). An LLM predicts in *token* space. JEPA predicts in *latent* space.

**How it works (for a diagram):**
1. Split the input into a *context* (visible) part and a *target* (masked) part.
2. A context encoder embeds the visible part.
3. A target encoder embeds the hidden part. It is a slowly updated copy of the context encoder, which stops the model from "cheating" by mapping everything to the same vector (called collapse).
4. A predictor tries to produce the target embedding from the context embedding.
5. The loss is the distance between the predicted and actual embeddings, *not* pixel or token reconstruction.

**Timeline:**
- 2022: LeCun's position paper, *A Path Towards Autonomous Machine Intelligence*, proposes JEPA as part of a larger agent design (world model, cost module, planner). The video is already on slide 23.
- 2023: **I-JEPA** (images). Assran et al., CVPR 2023.
- 2024: **V-JEPA** (video). Bardes et al.
- June 2025: **V-JEPA 2**. About 1.2B parameters, trained on more than 1M hours of video. **V-JEPA 2-AC** (action-conditioned) planned simple robot pick-and-place in new environments using a small amount of robot data. ⚠️ Check the exact robot-data figure in arXiv 2506.09985.
- ⚠️ 2025: **LLM-JEPA**. Applies a JEPA-style objective alongside next-token training for language (Huang, LeCun, Balestriero). My recollection only; search didn't confirm.
- ⚠️ Late 2025 / 2026: **VL-JEPA** (vision-language). Reported to predict continuous embeddings instead of generating tokens, with fewer parameters and much less training data than comparable models. Sources disagree on date and size.
- ⚠️ Nov 2025: LeCun leaves Meta. ⚠️ March 2026: founds **AMI Labs** (Paris), about $1.03B seed. A **V-JEPA 2.1** release is reported alongside.

**World models more broadly** (could be a sub-slide):
- *Generative* world models, such as DeepMind's **Genie 3** (Aug 2025, interactive environments you can move through in real time) and NVIDIA **Cosmos** (Jan 2025, physics-aware video for robotics), predict pixels.
- JEPA-style models predict latents. This contrast is the debate.
- **Energy-based models:** LeCun frames JEPA as scoring how *compatible* two things are, rather than generating one from the other. Worth one sentence, not a slide.

**Honest framing for the slide:** JEPA is promising for video, robotics, and planning. It has *not* yet produced a general-purpose chat model that competes with LLMs. Present it as a bet, with a prominent, well-funded advocate.

**Resources to verify:**
- [JEPA paper explained (Yannic Kilcher)](https://www.youtube.com/watch?v=jSdHmImyUjk) (already in deck)
- V-JEPA 2 paper: arXiv 2506.09985
- Meta AI blog posts for I-JEPA and V-JEPA
- ⚠️ [LeCun JEPA timeline (StartupHub)](https://www.startuphub.ai/news/figure-yann-lecun-jepa-release-cadence-2026-08-20) (secondary)
- ⚠️ [AMI Labs seed round (ai2.work)](https://ai2.work/blog/ami-labs-record-1-03b-seed-signals-the-world-model-revolution) (secondary)

### F3. Efficient Sequence Models: SSMs, Mamba, and Hybrids

**Core idea:** Attention compares every token with every earlier token, so cost grows with the square of length (Slide 9). **State space models (SSMs)** carry a fixed-size running state, like an RNN (Slide 5), but are designed to train in parallel. Cost grows linearly with length, and memory per generated token is constant.

**Teaching hook:** This is the RNN coming back. It brings Slides 5–6 full circle: "RNNs couldn't train in parallel; SSMs fixed that."

**Key points:**
- **Mamba** (Gu & Dao, Dec 2023): a *selective* SSM. It decides at each step what to keep in its state. **Mamba-2** (2024) showed a close mathematical link to attention.
- **RWKV**: an RNN that trains like a Transformer; an active open-source community. **RetNet** (Microsoft, 2023): mainly a research design.
- **Linear attention / DeltaNet family**: attention reformulated to avoid the square-law cost.
- **The catch:** a fixed-size state can't remember everything. Pure SSMs do worse at exact recall ("copy the phone number from 50 pages ago"). That's why the industry moved to **hybrids**, which keep a few full-attention layers for precise lookup.
- **Hybrid examples:** AI21 **Jamba** (2024: attention + Mamba + MoE); NVIDIA **Nemotron-H** (2025); **Qwen3-Next** (Sept 2025: mostly linear-attention layers with periodic full attention, plus MoE); MiniMax **M1** (2025, "lightning attention"). ⚠️ Check for newer 2026 hybrids.

**Analogy idea:** Attention is re-reading your entire notebook before writing each word. An SSM keeps a running summary on an index card. Hybrids keep the index card *and* occasionally flip back through the notebook.

**Resources:** the Mamba videos already on slide 23 (Yannic Kilcher; Maarten Grootendorst); Maarten Grootendorst's "A Visual Guide to Mamba and State Space Models" (article; verify URL).

### F4. Mixture of Experts: Big Models, Small Bills

**Core idea:** In a dense model, every parameter runs for every token. In an MoE model, each feed-forward block (Slide 2 / Slide 8) is replaced by many smaller "expert" feed-forward networks plus a **router**, which picks a few experts per token.

**Teaching hook:** It's the feed-forward block from Slide 2, multiplied. This is also why "model size" on the Recap slide needs a footnote: **total vs. active parameters**.

**Key points:**
- **DeepSeek-V3** (Dec 2024): 671B total, about 37B active per token. ⚠️ DeepSeek V4 reported at about 1.6T total / 49B active (secondary source, verify). Other examples: **Llama 4**, **Mixtral**, **Qwen3 MoE**, **Gemini** (Google has stated it uses sparse MoE).
- **The misconception to fix:** "experts" are not subject specialists (a "math expert," a "French expert"). Routing happens per token, in every layer, and experts tend to specialize in low-level patterns like punctuation or particular word pieces.
- **Trade-offs:** cheaper compute per token, but *all* experts must sit in memory. That makes serving expensive and engineering harder (load-balancing the router).
- Many leading open-weight models are now MoE.

**Analogy idea:** A hospital with 64 specialists where each patient sees only 2. You pay 2 salaries per visit, but you still need a building big enough for all 64.

**Resources:** IBM "What is Mixture of Experts?" (already in deck); Maarten Grootendorst's "A Visual Guide to Mixture of Experts" (verify URL); DeepSeek-V3 technical report (arXiv 2412.19437).

### F5. Generating Differently: Diffusion LMs, Multi-Token, and Concept Models

**Core idea:** Every LLM in the deck writes one token at a time (Slide 12). Three research lines challenge that.

1. **Diffusion language models.** Borrowed from image diffusion (now mentioned on Slide 18). Start from a fully masked or noisy sequence and refine *all positions in parallel* over a number of steps.
   - **LLaDA** (Feb 2025, open weights, 8B): a masked-diffusion LLM.
   - **Mercury** (Inception Labs, 2025): commercial diffusion LLM, very fast. ⚠️ **Mercury 2** reported in 2026.
   - **Gemini Diffusion** (Google DeepMind, I/O May 2025): experimental; demoed at roughly 1,000–2,000 tokens/sec.
   - Trade-offs: large speed gains and the ability to revise earlier text. Quality on some tasks and long-form reasoning has lagged. Speed figures are vendor-reported. [Overview](https://www.web3aiblog.com/blog/diffusion-llms-explained-mercury-gemini-diffusion-2026) · [Deployment notes](https://www.spheron.network/blog/deploy-diffusion-language-models-dllm-gpu-cloud-2026/)
2. **Multi-token prediction.** Train the model to predict several upcoming tokens at once (Meta, Gloeckle et al., 2024). **DeepSeek-V3** used it as a training aid and to speed up generation. Pairs naturally with **speculative decoding** (a small model drafts, the big model checks).
3. **Large Concept Models (LCM)** (Meta FAIR, Dec 2024). Predict the next *sentence embedding* (in a language-independent sentence space called SONAR) instead of the next token. Philosophically close to JEPA: reason over ideas, then render words. Possibly the intended "NCM."

**Analogy idea:** Autoregressive generation is typing left to right. Diffusion is sketching the whole page faintly and then sharpening it everywhere at once.

### F6. Thinking Without Words: Latent and Recursive Reasoning

**Core idea:** Slide 20 says reasoning happens by writing out tokens. What if the model could think in vectors instead?

- **Coconut** ("Chain of Continuous Thought," Meta, Dec 2024): feeds the model's last hidden state back in as the next input *without* turning it into a token. The model can then keep several possibilities open instead of committing to one word.
- **Small recursive models:** **HRM** (Hierarchical Reasoning Model, Sapient, mid-2025, about 27M parameters) and **TRM** (Tiny Recursive Model, Samsung, Oct 2025, about 7M parameters). Both loop a tiny network many times over its own internal state, and both scored surprisingly well on puzzle benchmarks like ARC-AGI and Sudoku. ⚠️ Verify scores; there has been debate about what drives the results.
- **Trade-off to teach:** latent reasoning is potentially more efficient but **less inspectable**. That connects to the faithfulness caveat now on Slide 20, and to interpretability (F8).

### F7. Models That Keep Learning: Test-Time Memory and Continual Learning

**Core idea:** The deck's central mental model is "frozen weights" (Slides 1, 10, 22). This slide shows the research trying to break that rule, which makes it a strong closing idea.

- **Test-time training:** update a small part of the model's weights *while* processing the input.
- **Titans** (Google, Behrouz et al., Dec 2024 / Jan 2025): adds a neural long-term memory module that learns during inference, prioritizing what is "surprising."
- **Nested Learning / HOPE** (Google, NeurIPS 2025): treats a model as layers of learning that update at different speeds (a "continuum memory system"), aimed at continual learning without forgetting. ⚠️ Verify details.
- **The hard problem: catastrophic forgetting.** Learning new things tends to overwrite old ones. That's a big reason production models are still frozen and use RAG and memory files instead (Slides 10, 17, 21).
- ⚠️ Check whether any 2026 production model ships genuine weight updates during use. As of mid-2026, "memory" features were still prompt-based.

### F8. Looking Inside (optional) and Beyond Tokenizers (optional)

These two are smaller. Either combine them or attach them to earlier slides as callouts.

- **Mechanistic interpretability.** This closes the loop on Slide 4's superposition fix.
  - Anthropic's *Toy Models of Superposition* (2022), [transformer-circuits.pub](https://transformer-circuits.pub/2022/toy_model/index.html).
  - *Scaling Monosemanticity* (2024): sparse autoencoders found millions of readable features in Claude 3 Sonnet. This led to the "Golden Gate Claude" demo.
  - *On the Biology of a Large Language Model* / circuit tracing (2025): traced how a model plans rhymes and does multi-step reasoning internally.
  - Teaching value: it shows that we *can* partly see the "directions of meaning."
- **Beyond tokenizers.** This closes the loop on Slide 3.
  - **Byte Latent Transformer** (Meta, Dec 2024) reads raw bytes and groups them into variable-size patches based on how predictable the next byte is. That removes the fixed tokenizer and its character blindness.

---

## Implementation notes for the build session

- **Category:** A new `frontier` category would need a CSS color variable and `li::before` rule in `model-basics.css` (around line 12 and lines 274–277), and the `validCategories` list in `tests.html` (line 136). The alternative is to keep `adv`.
- **Slide count:** Going from 23 to roughly 29–30 means updating the count and topic list in `README.md` and `CLAUDE.md`, per the pre-commit alignment rule.
- **Version:** bump `index.html` (currently v26).
- **Order:** decide whether the Recap (slide 22) moves to the end.
- **Images:** JEPA, MoE, and diffusion each benefit greatly from a diagram. Find properly licensed figures (Meta blog, Maarten Grootendorst's visual guides with attribution, or Wikimedia).
- **Link check:** after adding resources, rerun the YouTube title check used in the v26 pass, which queries `youtube.com/oembed?url=...`. Two dead links and four mislabeled ones were found that way.

---

## Cheat-sheet (`docs/cheat-sheet.yaml`) sync checklist

The v26 pass corrected these points in `cards-data.js`. The cheat sheet should be checked for each:

- [ ] **Causal mask:** decoder-only tokens attend only to earlier tokens. Add as a term.
- [ ] **KV cache / prefill vs. decode:** the prompt is read once in parallel, then one pass per generated token. Nothing is re-tokenized.
- [ ] **Residual stream:** add as a term (fits the skyscraper/elevator analogy).
- [ ] **Embeddings:** meaning lives in *directions*, not single dimensions (superposition). The lookup is context-free; layers add context. King − Man + Woman ≈ Queen (whole-vector offset). The cheat sheet's "typically 512–4096 dimensions" (line 38) is fine, but large models now go well beyond 4,096.
- [ ] **Positional encoding:** RoPE inside attention for most modern LLMs.
- [ ] **Temperature:** divides logits; 1.0 = unchanged; 0 = greedy. Add **top-k / top-p**.
- [ ] **Context window overflow:** the application trims or summarizes; the model doesn't "forget."
- [ ] **Chat template:** the prompt stack is one token sequence with role markers.
- [ ] **Post-training:** add RLAIF / Constitutional AI and **RLVR** (how reasoning models are trained). Base models don't refuse; they just continue text. Sycophancy as a side effect.
- [ ] **Reasoning models:** same architecture, different training; thinking tokens are ordinary tokens; faithfulness caveat.
- [ ] **RAG:** embedding-based retrieval mechanism; *not* a privacy guarantee with hosted models; access-control and prompt-injection risks (cheat-sheet line ~320 mentions "data privacy and security").
- [ ] **Multimodal:** vision encoder + adapter; CLIP contrastive alignment; generation is mostly diffusion.
- [ ] **Tool use:** a tool call is generated text; the app executes it; MCP.
- [ ] **Agents:** repeated tool use; memory = context + external notes.
- [ ] **Fine-tuning:** changes behavior and style, poor at adding facts.
- [ ] **MoE:** experts are not subject specialists; total vs. active parameters.
- [ ] **Emerging architectures** (cheat-sheet line ~331): rebuild from the Frontier section once it's written.
