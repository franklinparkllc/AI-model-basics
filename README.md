# AI Model Basics

An interactive slideshow that teaches 21 core concepts about how modern AI models are built, trained, and used — from tokenization and transformers to RAG, reasoning, and agentic workflows.

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
2. Tokens & Tokenization
3. From Numbers to Meaning: Embeddings
4. Feed-Forward Neural Networks
5. Recurrent Neural Networks (RNNs)
6. Long Short-Term Memory (LSTM)
7. Encoder-Decoder Architecture (Seq2Seq)
8. The Transformer Architecture
9. How Attention Works

### Training
10. Pre-Training
11. Post-Training
12. Bias, Fairness & Limitations

### Inference
13. The Frozen State & Prompt Stack
14. What Happens When You Send a Message?
15. The Selection Dice Roll

### Advanced
16. RAG: Giving Models Access to Knowledge
17. Beyond Text: Multimodal Models & Tool Use
18. Reasoning: Two Paradigms
19. Agentic Workflows
20. Looking Ahead — Emerging Architectures

### Conclusion
21. Recap: Key Learnings

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
