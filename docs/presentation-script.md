# Presentation Materials: The Architecture of Modern AI

## Part 1: The 15-Minute Script

### 0:00 – 2:00 | The Engine Overview

Good afternoon. We often talk about AI as if it's magic, but it is actually a highly engineered mathematical engine. Today, we are going to peel back the hood and look at the architecture, the training, and the inference—the process of how we turned trillions of words into a functional, "frozen" digital brain.

The turning point for everything we see today was 2017. A research paper titled "Attention is All You Need" introduced the Transformer architecture. Before this, AI was often limited to narrow, specific tasks. After this, it became a generalist. This architecture unlocked the ability to scale models to billions—and now trillions—of parameters.

### 2:00 – 4:30 | Tokens and the Map of Meaning

Before the engine can run, it needs fuel. Computers do not read letters; they process numbers. This is Tokenization. When you type a word like "Ingenious," the model breaks it into fragments: In, gen, and ious. This explains why AI sometimes struggles with spelling or counting letters—it isn't seeing the word; it's seeing a sequence of numeric IDs.

Once we have those IDs, we need to give them meaning. In AI, every token is assigned a "scouting report" called an Embedding. This is a list of thousands of numbers—Dimensions—that represent traits. The model discovers these traits organically during training. These embeddings live in Latent Space, a high-dimensional coordinate system where similar concepts cluster together. In this space, "Apple" the fruit and "Orange" sit physically close to each other, while "Apple" the technology company lives in a different neighborhood entirely.

Before the first layer, the model adds Positional Encoding—patterns that tell it where each token sits in the sequence. That step is what lets self-attention know order: without it, "dog bites man" and "man bites dog" would look identical. So the pipeline is: tokenize, embed, add position, then the stack of layers. Once that structure is set, the prompt is like a fixed-shape array: the same number of tokens and dimensions from bottom to top. Through every layer, only the numbers in those vectors change—attention and feed-forward update the values, not the shape.

### 4:30 – 8:30 | The Memory Crisis and the Attention Revolution

The biggest hurdle in AI history was Memory. Language is sequential; the meaning of a word at the end of a sentence often depends on a word at the very beginning.

For years, we used a system called Baton-Passing (RNNs). The model would read the first word, summarize it, and pass that summary to the next word. But in a long sentence, the "runner" got tired. By the time the model reached the end of a paragraph, it had dropped the baton and forgotten how it started.

In 2014, researchers tried to fix this by giving the runner Binoculars. This was early Attention. It allowed the model to "look back" at the beginning of the sentence when it got stuck. But the 2017 breakthrough was realizing that if the binoculars were powerful enough, you didn't need the runner at all. The Transformer replaced the relay race with a massive, simultaneous Search. It uses three vectors:

- The Query (Q): "What am I looking for?"
- The Key (K): "What do I represent?"
- The Value (V): "What information do I carry?"

### 8:30 – 11:30 | The Skyscraper and Training

A Transformer is a Skyscraper of these search layers. Every floor has an Attention Station, where tokens talk to each other to gather context, and a Feed-Forward Station, where the model refines that information. As data rides the elevator to the top, it becomes increasingly nuanced.

How does it learn? First, through Pre-training, where it reads the internet to learn "predict the next word." This creates a Base Model—a brilliant but unruly generalist. Then comes Post-training (like RLHF or DPO), which uses human feedback to turn that generalist into a helpful, safe Assistant.

### 11:30 – 15:00 | Inference and Advanced Reasoning

Once training is done, the weights are Frozen. When you chat with a model, you aren't teaching it; you are providing temporary context in a Prompt Stack. At the "roof" of the skyscraper, the model generates probabilities for the next word. Temperature is the dice roll—low temperature for predictability, high temperature for creativity.

To overcome the fact that the model is frozen in time, we use RAG (Retrieval-Augmented Generation). This is an "Open Book Exam" where the model looks at your specific documents to find facts. Finally, we are entering the era of Reasoning Models, which use "Inference-Time Compute" to pause and think, exploring different solutions before they speak.
