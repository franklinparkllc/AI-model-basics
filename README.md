# AI Model Basics

An interactive slideshow that teaches 23 concepts about how modern AI models are built, trained, and used — from tokenization and transformers to RAG, reasoning, and agentic workflows.

Each slide includes clear explanations with analogies, key bullet points, visual callouts, and curated external resources.

## Getting Started

Open `index.html` directly in any modern web browser. No build process or dependencies required.

**Navigation:**
- Arrow keys (← →)
- On-screen navigation buttons
- Touch/swipe (mobile)
- Direct URL hash (e.g., `#slide-5`)

## Topics Covered

### Architecture
1. Introduction to Modern AI Models
2. Feed-Forward Neural Networks (FFNN)
3. Tokens & Tokenization
4. From Numbers to Meaning: Embeddings & Latent Space
5. Recurrent Neural Networks (RNNs)
6. Long Short-Term Memory (LSTM)
7. Encoder-Decoder Architecture (Seq2Seq)
8. The Transformer Architecture
9. How Attention Works

### Inference
10. The Frozen State & Prompt Stack
11. Preparing the Input
12. The Forward Pass & Autoregressive Loop
13. The Selection Dice Roll

### Training
14. Pre-Training
15. Post-Training
16. Bias, Fairness & Limitations

### Advanced
17. RAG & Data Security
18. Multimodal Models: Richer Perception
19. Tool Use: From Words to Actions
20. Reasoning: Two Paradigms
21. Agentic Workflows

### Conclusion
22. Recap: Key Learnings
23. Looking Ahead — Emerging Architectures

## File Structure

```
AI-model-basics/
├── index.html          # Main entry point; nav, version badge, script tags
├── model-basics.js     # Slideshow controller (navigation, rendering, hash routing)
├── cards-data.js       # All slide content as a JavaScript array
├── model-basics.css    # All styling (CSS Grid layout, transitions, responsive)
├── tests.html          # In-browser test suite (data validation + slideshow logic)
├── docs/
│   ├── cheat-sheet.html    # Printable/shareable reference sheet
│   └── cheat-sheet.yaml    # Source data for the cheat sheet
└── README.md
```

## Features

- **No build step** — pure HTML/CSS/JS, zero dependencies
- **Progress bar** — visual indicator of position through the deck
- **Version badge** — displayed in the nav; incremented with each content update
- **AI Timeline link** and **Cheat Sheet** accessible from the nav bar
- **Responsive** — works on desktop, tablet, and mobile

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- iOS Safari, Chrome Mobile

## License

Open source, available for educational purposes.
