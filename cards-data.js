// Model Basics - Card Data
// All content for the topic cards
// Incorporating "Factory/Skyscraper" and "Flavor Profile" analogies

const cardsData = [
    {
        category: 'arch',
        badge: 'Overview',
        title: '1. Introduction to modern AI models',
        description: 'AI models are sophisticated mathematical engines that have seen remarkable growth in the last decade',
        paragraphs: [
            'In this presentation, we are going to examine the mechanisms of how modern AI systems work.',
            'Broadly speaking, there are three main topics to cover: model architecture, model training, and model inference.',
            'At the heart of modern models is a concept called the <strong>Transformer</strong>, which is a type of neural network architecture that is designed to process text data. Transformers were defined in a seminal paper in 2017 by Vaswani et al. and have since become the de facto standard for language model architecture. Transformers unlocked the ability to train models with billions of parameters, which is what allows modern models to be so powerful.'
        ],
        bullets: [
            'Understanding this architecture helps predict failures, hallucinations, and informs effective use of tools like RAG',
            'The pipeline flows: <strong>Architecture</strong> → <strong>Training</strong> → <strong>Inference</strong> → <strong>Advanced Capabilities</strong>'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Training vs. Inference:</strong> Training updates the model\'s "brain" (weights)—a massive structure that can contain billions—sometimes hundreds of billions—of parameters. Inference is the act of querying that "frozen" brain. Chatting provides temporary context, but it does not permanently teach the model or update its knowledge base.'
        },
        resources: [
            { type: 'video', title: 'Generative AI in a Nutshell', meta: '18 min · Henrik Knibbe', url: 'https://www.youtube.com/watch?v=2IK3DFHRFfw' },
            { type: 'article', title: 'A Short History of Neural Networks', meta: '20 min · David D. Nolte', url: 'https://galileo-unbound.blog/2025/02/05/a-short-history-of-neural-networks/' },
            { type: 'interactive', title: 'OKAI — Interactive Intro to AI', meta: 'Brown University', url: 'https://okai.brown.edu/' },
            { type: 'video', title: 'Large Language Models, briefly', meta: '8 min · 3Blue1Brown', url: 'https://www.youtube.com/watch?v=LPZh9BOjkQs' }
        ],
        image: {
            url: 'https://upload.wikimedia.org/wikipedia/commons/8/8f/The-Transformer-model-architecture.png',
            caption: 'The Transformer model architecture',
            attribution: 'Vaswani et al., 2017'
        }
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '2. Tokens & Tokenization',
        description: 'Models don\'t read text directly—they process numeric token IDs that represent pieces of words.',
        paragraphs: [
            'AI models operate on numbers, not letters. A <strong>tokenizer</strong> converts text into integer IDs representing vocabulary fragments. For example, "Ingenious" might split into three tokens: <code>In</code>, <code>gen</code>, and <code>ious</code>.',
            'This approach (often <strong>BPE</strong>—<strong>Byte Pair Encoding</strong>—or similar) balances efficiency and flexibility: common words stay whole, rare words split into reusable parts.',
            'Modern implementations like GPT\'s tokenizer use byte-level BPE, which operates on raw bytes rather than characters, using the 256 possible byte values (0-255) as a universal foundation. This allows it to represent any language, emoji, or Unicode character through UTF-8 encoding, making it language-agnostic.',
            'Multimodal models do the same idea for images (patches) and audio (chunks).'
        ],
        bullets: [
            'Token count determines cost and speed—more tokens = higher compute',
            'Tokenization explains quirks: spelling/backwards tasks are hard (tokens don\'t map 1:1 to letters)',
            '<strong>Tokens are pieces, not words:</strong> The model often sees subwords like <code>un</code> + <code>believ</code> + <code>able</code>',
            'BPE originated in data compression (1994) and was later adopted for NLP tokenization; modern tokenizers repurpose that idea.'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Key idea:</strong> Most language models are trained on one core objective: <em>predict the next token</em>. That single skill can look like reasoning, writing, or coding—but it\'s still prediction, not guaranteed "truth" or perfect calculation.'
        },
        resources: [
            { type: 'tool', title: 'OpenAI Tokenizer', meta: 'Interactive · Try BPE', url: 'https://platform.openai.com/tokenizer' },
            { type: 'video', title: 'Byte Pair Encoding Explained', meta: '7 min · Tokenization', url: 'https://www.youtube.com/watch?v=4A_nfXyBD08' },
            { type: 'video', title: 'Build GPT Tokenizer', meta: '2h 13min · Andrej Karpathy', url: 'https://www.youtube.com/watch?v=zduSFxRajkE' },
            { type: 'article', title: 'The Art of Tokenization', meta: 'Towards Data Science', url: 'https://towardsdatascience.com/the-art-of-tokenization-breaking-down-text-for-ai-43c7bccaed25/' }
        ],
        image: {
            url: 'https://towardsdatascience.com/wp-content/uploads/2024/09/1QVXvydRMEWTWiUP42bYBAg.png',
            caption: 'Tokenization visualization',
            attribution: 'Towards Data Science'
        }
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '3. From Numbers to Meaning: Embeddings & Latent Space',
        description: 'Token IDs are just numbers—embeddings transform them into rich, meaningful representations in a high-dimensional "Latent Space."',
        paragraphs: [
            'After tokenization, the model has a sequence of token IDs—integers like [4829, 2121, 8945]. But numbers alone are meaningless. The model needs to understand what each token <em>represents</em>.',
            'Enter <strong>embeddings</strong>: the model looks up each token ID in a learned table (the <strong>embedding matrix</strong>) and retrieves a <strong>vector</strong>—a list of hundreds or thousands of numbers. Each dimension captures some aspect of meaning: semantic properties, grammatical role, contextual patterns. This idea—that meaning comes from context ("you shall know a word by the company it keeps," 1957)—led from early methods like Word2Vec (2013) to the powerful embeddings in modern Transformers.',
            'These vectors live in <strong>Latent Space</strong>, a high-dimensional coordinate system where similar meanings cluster together. A vector might have 4,096 dimensions, each representing a "latent" feature the model discovered during training—even if humans don\'t have a name for it.',
            'Finally, the model adds <strong>positional encodings</strong> so it knows word order. Without them, "dog bites man" and "man bites dog" would look identical. Position + meaning = the full input representation that flows into the Transformer layers.'
        ],
        bullets: [
            '<strong>Embedding Lookup:</strong> Token ID → retrieve learned vector from embedding table',
            '<strong>Latent Space:</strong> An organic "map of meaning" where distance = difference in concept',
            '<strong>Learned Features:</strong> Models discover their own dimensions of meaning during training—not hand-crafted by humans',
            '<strong>Positional Encoding:</strong> Adds order information so word sequence matters; applied after embed, before first Transformer layer'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>The Flavor Profile Analogy:</strong> If tokenization assigns each word a locker number, embeddings are the contents—a profile describing the word\'s "flavor." One dimension might be "sweetness," another "spiciness." In a 4,096-dimensional space, the model creates a hyper-detailed profile for every concept.'
        },
        resources: [
            { type: 'video', title: 'Tokens and Embeddings', meta: '7 min · Visual', url: 'https://www.youtube.com/watch?v=izbifbq3-eI' },
            { type: 'video', title: 'Language Models & Transformers', meta: '20 min · Computerphile', url: 'https://www.youtube.com/watch?v=rURRYI66E54' },
            { type: 'article', title: 'An Introduction to Embeddings', meta: 'DDBM', url: 'https://www.ddbm.com/en/blog/an-introduction-to-embeddings' }
        ],
        image: {
            url: 'https://www.ddbm.com/hs-fs/hubfs/Imported_Blog_Media/linear-relationships-4-1.jpg?width=1534&name=linear-relationships-4-1.jpg',
            caption: 'Embeddings in latent space',
            attribution: 'DDBM'
        }
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '4. Feed-Forward Neural Networks (FFNN)',
        description: 'The "classic" neural network architecture where data flows in one direction—the ancestor of all modern AI.',
        paragraphs: [
            'Rooted in the <strong>Perceptron</strong> (1958), feed-forward neural networks process information in one direction—input → hidden layers → output—with no loops or memory. The key breakthrough was <strong>backpropagation</strong> (1986), which enabled training of deep, multilayer networks capable of learning complex, non-linear relationships.',
            'FFNNs are the foundation of deep learning, but they process each input independently. Because they have no memory of previous inputs, they struggle with sequential data like text or audio—a limitation that motivated the development of RNNs, and eventually the Transformer\'s Attention mechanism.',
            'A key ingredient is the <strong>activation function</strong> (e.g., ReLU) applied between layers. Without it, stacking layers would just produce another linear transformation—no matter how deep. Activation functions introduce <strong>non-linearity</strong>, allowing the network to learn complex, curved decision boundaries instead of just straight lines.'
        ],
        bullets: [
            '<strong>One-Way Flow:</strong> Data flows forward only (input → output), no feedback loops',
            '<strong>No Memory:</strong> Each input processed independently, no context from previous inputs',
            '<strong>Backpropagation (1986):</strong> Enables training of deep multilayer networks',
            '<strong>Activation Functions:</strong> Introduce non-linearity so the network can learn complex patterns, not just linear relationships',
            '<strong>Limitation:</strong> Cannot model temporal dependencies or sequential patterns'
        ],
        callout: {
            type: 'insight',
            content: '<strong>The Foundation:</strong> Feed-forward networks established the basic architecture and training methods (backpropagation) that all subsequent neural networks build upon. Their limitation—no memory—led to the development of RNNs for sequential data.'
        },
        resources: [
            { type: 'video', title: 'The Essential Main Ideas of Neural Networks', meta: '19 min · StatQuest', url: 'https://youtu.be/CqOfi41LfDw?si=vGamzRxa1mtcQ3nf' },
            { type: 'video', title: 'Feed Forward Neural Network (FFNN)', meta: '14 min · Kenan Casey · Machine Learning Distilled', url: 'https://www.youtube.com/watch?v=VZ-TvUvtDbg' }
        ],
        image: {
            url: 'https://media.geeksforgeeks.org/wp-content/uploads/20251209120638608023/bhu.webp',
            caption: 'A perceptron diagram',
            attribution: 'GeeksForGeeks'
        }
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '5. Recurrent Neural Networks (RNNs)',
        description: 'RNNs extend feed-forward networks with memory, enabling them to process sequential data.',
        paragraphs: [
            '<strong>Recurrent Neural Networks (RNNs)</strong> were developed in 1985 to solve the fundamental limitation of feed-forward networks: their inability to handle sequential data. RNNs introduce feedback loops, allowing information to flow backward and creating a "memory" mechanism.',
            'Unlike feed-forward networks, RNNs maintain a <strong>hidden state</strong> that carries information from previous time steps. At each step, the RNN takes the current input and the previous hidden state, combines them, and produces both an output and a new hidden state. This allows the network to remember and use information from earlier in the sequence.',
            'RNNs excel at tasks where order matters: language modeling, speech recognition, time series prediction. However, they have a critical flaw: the <strong>vanishing gradient problem</strong>. When processing long sequences, gradients shrink exponentially as they propagate backward through time. By the time an RNN reaches word 50, it has largely forgotten word 1, limiting their ability to capture long-range dependencies.'
        ],
        bullets: [
            '<strong>Sequential Processing:</strong> Must process tokens one at a time, cannot parallelize',
            '<strong>Hidden State:</strong> Carries context from previous steps forward',
            '<strong>Memory:</strong> Can remember information from earlier in the sequence',
            '<strong>Vanishing Gradients:</strong> Information decays over long sequences'
        ],
        callout: {
            type: 'insight',
            content: '<strong>The Trade-off:</strong> RNNs gained memory at the cost of parallelization. They must process sequences sequentially, which is inherently slow and cannot leverage parallel GPU computation—a bottleneck that would eventually limit model scale.'
        },
        resources: [
            { type: 'video', title: 'Recurrent Neural Networks (RNN) - Clearly Explained', meta: '16 min · StatQuest', url: 'https://www.youtube.com/watch?v=AsNTP8Kwu80' },
            { type: 'article', title: 'Understanding LSTM Networks', meta: 'Chris Olah · Intro covers RNNs', url: 'https://colah.github.io/posts/2015-08-Understanding-LSTMs/' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '6. Long Short-Term Memory (LSTM)',
        description: 'LSTMs improve on RNNs with gating mechanisms that better preserve long-term memory.',
        paragraphs: [
            '<strong>Long Short-Term Memory (LSTM)</strong> networks were designed to solve RNNs\' vanishing gradient problem. They introduce a more sophisticated memory mechanism with three gates: <strong>forget gate</strong> (what to discard), <strong>input gate</strong> (what to store), and <strong>output gate</strong> (what to use).',
            'The key innovation is the <strong>cell state</strong>—a highway that runs through the entire sequence, allowing information to flow relatively unchanged. The gates control what information enters, stays, or exits this cell state.',
            'LSTMs significantly improved the ability to capture long-range dependencies, but they still suffer from the same fundamental limitation: <strong>sequential processing</strong>. They must process tokens one at a time, preventing parallelization and limiting scalability.'
        ],
        bullets: [
            '<strong>Gating Mechanisms:</strong> Forget, input, and output gates control information flow',
            '<strong>Cell State:</strong> Highway for long-term information preservation',
            '<strong>Better Memory:</strong> Can capture dependencies over longer sequences than RNNs',
            '<strong>Still Sequential:</strong> Cannot parallelize, remains a bottleneck'
        ],
        callout: {
            type: 'note',
            content: '<strong>Improvement, Not Solution:</strong> LSTMs improved memory but didn\'t solve the parallelization problem. They still process sequences sequentially, which limits their scalability.'
        },
        resources: [
            { type: 'video', title: 'LSTM Networks - Clearly Explained', meta: '20 min · StatQuest', url: 'https://www.youtube.com/watch?v=YCzL96nL7j0' },
            { type: 'article', title: 'Understanding LSTM Networks', meta: 'Chris Olah · Classic explainer', url: 'https://colah.github.io/posts/2015-08-Understanding-LSTMs/' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '7. Encoder-Decoder Architecture (Seq2Seq)',
        description: 'The "Seq2Seq" breakthrough introduced the "Thought Vector"—a way to turn a whole sentence into a single mathematical point.',
        paragraphs: [
            'RNNs and LSTMs process sequences, but they struggle with a key challenge: how do you transform one sequence into another sequence of different length? <strong>Encoder-decoder architecture</strong> (also called <strong>Seq2Seq</strong>) was invented in 2014 by Ilya Sutskever and Kyunghyun Cho to solve this.',
            'The architecture splits the task between two RNNs: The <strong>encoder</strong> processes the entire input sequence, compressing it into a single fixed-size vector in Latent Space called the <strong>context vector</strong> (often called a "Thought Vector"). The <strong>decoder</strong> then takes this single vector and "unpacks" it into a new sequence.',
            '<strong>Terminology note:</strong> "Thought Vector" was popular in early Seq2Seq literature but modern papers typically use "context vector" or "hidden state." If you search current research, look for "context vector" instead.',
            'Think of the "Thought Vector" as a <strong>universal coordinate</strong> for a concept. If you translate "The cat is on the mat" to French, the encoder finds the exact spot in the Map of Meaning (Latent Space) where that specific "thought" lives. The decoder then looks at that coordinate and describes it using French words.',
            'However, this revealed a critical limitation: the <strong>context bottleneck</strong>. Trying to compress a long, complex document into one single vector is like trying to summarize a whole book into one sentence—you lose the nuances. This limitation directly motivated the invention of <strong>Attention</strong>.'
        ],
        bullets: [
            '<strong>Context Vector:</strong> A single point in Latent Space representing an entire sequence\'s meaning',
            '<strong>Universal Coordinate:</strong> The same "thought" should sit in the same spot, regardless of the language used to describe it',
            '<strong>Variable-Length I/O:</strong> Finally allowed mapping different input and output lengths (e.g., translation)',
            '<strong>Context Bottleneck:</strong> A single fixed-size vector cannot hold infinite detail'
        ],
        callout: {
            type: 'insight',
            content: '<strong>The Bottleneck:</strong> Imagine trying to compress the entire plot of "War and Peace" into a single paragraph. You get the gist, but you lose the details. This "bottleneck" is what led researchers to ask: <em>What if the decoder could look back at the original words whenever it needed to?</em>'
        },
        resources: [
            { type: 'video', title: 'Sequence-to-Sequence (seq2seq), Clearly Explained', meta: '14 min · StatQuest', url: 'https://www.youtube.com/watch?v=L8HKweZIOmg' },
            { type: 'article', title: 'Visualizing A Neural Machine Translation Model', meta: '12 min · Jay Alammar', url: 'https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/' },
            { type: 'video', title: 'LSTM-Based Seq2Seq Explained', meta: '10 min · Encoder-Decoder', url: 'https://www.youtube.com/watch?v=L0sut0EL0mM' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '8. The Transformer Architecture',
        description: 'The breakthrough that eliminated sequential processing — attention without RNNs.',
        paragraphs: [
            'Attention was first added to Seq2Seq models (still RNN-based) to solve long-range dependencies. But if attention handles all context, <strong>why keep sequential RNNs at all?</strong> This question led to the <strong>Transformer</strong> (2017): an architecture built entirely on attention, processing all tokens in parallel.',
            'A Transformer is a <strong>stack of identical layers</strong> (12, 24, 48, 96+). Think of it as a skyscraper: each floor performs the same two operations on every token. (1) <strong>Self-attention</strong> lets tokens communicate and update based on context. (2) A <strong>feed-forward network</strong> processes each token independently, refining its meaning. Residual connections and layer normalization keep training stable across 100+ layers.',
            'Early layers learn grammar and syntax. Middle layers learn relationships. Deep layers encode reasoning and nuance. This <strong>progressive refinement</strong> is what makes Transformers so powerful — each layer adds a small improvement, but stacking dozens of them builds sophisticated understanding from simple token embeddings.',
            'Most modern LLMs (GPT, Claude) use <strong>decoder-only Transformers</strong> — just the generation half, without a separate encoder.'
        ],
        bullets: [
            '<strong>Layer Pattern:</strong> (Normalize → Attention → Add) → (Normalize → Feed-forward → Add)',
            '<strong>Parallel Processing:</strong> All tokens processed simultaneously — no sequential bottleneck',
            '<strong>Depth = Capability:</strong> More layers → more complex reasoning',
            '<strong>Decoder-Only:</strong> Most modern LLMs use only the decoder half'
        ],
        callout: {
            type: 'insight',
            content: '<strong>The Skyscraper Analogy:</strong> Ground floor tokens know only their own meaning. As they ride the elevator through dozens of floors — each adding context from surrounding words — they emerge at the top with rich, nuanced understanding of their role in the sentence.'
        },
        resources: [
            { type: 'video', title: 'Transformers, the tech behind LLMs', meta: '58 min · 3Blue1Brown', url: 'https://www.youtube.com/watch?v=KJtZARuO3JY' },
            { type: 'article', title: 'The Illustrated GPT-2', meta: 'Jay Alammar · Visualizing GPT-2', url: 'https://jalammar.github.io/illustrated-gpt2/' },
            { type: 'article', title: 'Attention? Attention!', meta: '21 min · Lilian Weng', url: 'https://lilianweng.github.io/posts/2018-06-24-attention/' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '9. How Attention Works',
        description: 'Inside each Transformer layer, self-attention lets every token figure out which other tokens matter most — using three simple vectors.',
        paragraphs: [
            'Remember the self-attention step inside each Transformer layer (Slide 8)? Here\'s how it actually works. Each layer has learned weight matrices that transform every token\'s embedding into three new vectors: <strong>Query (Q)</strong> — "What am I looking for?", <strong>Key (K)</strong> — "What information do I have?", and <strong>Value (V)</strong> — "What content should I contribute?"',
            'Every token\'s Query is compared against every other token\'s Key (via dot product) to produce <strong>attention scores</strong> — essentially a relevance ranking. <strong>Softmax</strong> then converts these raw scores into probabilities that sum to 1. High-relevance tokens get most of the weight; irrelevant tokens get nearly zero. The final output is a weighted blend of all tokens\' Values.',
            '<strong>Example:</strong> In "The bank by the river," the word "bank" sends out its Query. When matched against Keys, "river" scores high while "the" scores low. So "bank" pulls mostly from "river\'s" Value, nudging its representation toward "riverbank" rather than "financial institution." This happens for every token simultaneously.',
            'Models don\'t run this process once — they run it <strong>8-32 times in parallel</strong>, each with its own set of Q/K/V weight matrices. Each "head" learns to focus on different relationships: one might track grammar, another semantics, another pronoun references. The outputs are concatenated and mixed, giving the model multiple simultaneous perspectives on every token.'
        ],
        bullets: [
            '<strong>Learned Weights:</strong> Each layer\'s Q, K, V matrices are trained parameters — the model learns <em>what to pay attention to</em>',
            '<strong>Softmax:</strong> Converts raw scores to probabilities summing to 1 — this same function reappears at inference (Slide 15) to select output tokens',
            '<strong>Multi-Head:</strong> 8-32 parallel attention operations, each specializing in different patterns',
            '<strong>All At Once:</strong> Every token attends to every other token simultaneously — no sequential bottleneck'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>The Cocktail Party:</strong> Imagine you\'re at a crowded party (self-attention). You tune into the conversations most relevant to you (high attention scores) and tune out background noise (low scores). Now imagine 16 versions of you at the same party, each listening for something different — one for names, one for emotions, one for topics. That\'s multi-head attention: multiple simultaneous filters on the same scene.<br><br><em>With the architecture understood, let\'s see how these structures actually learn.</em>'
        },
        resources: [
            { type: 'video', title: 'Attention in Transformers', meta: '26 min · 3Blue1Brown', url: 'https://www.youtube.com/watch?v=eMlx5fFNoYc' },
            { type: 'article', title: 'The Illustrated Transformer', meta: 'Jay Alammar · Multi-head', url: 'https://jalammar.github.io/illustrated-transformer/' },
            { type: 'video', title: 'Attention Is All You Need (walkthrough)', meta: '15 min · Visual', url: 'https://www.youtube.com/watch?v=wjZofJX0v4M' }
        ]
    },
    {
        category: 'train',
        badge: 'Training',
        title: '10. Pre-Training',
        description: 'Pre-training is where models learn the patterns, facts, and structures of human knowledge from massive text datasets.',
        paragraphs: [
            'During pre-training, the model consumes <strong>trillions of tokens</strong> from books, websites, research papers, and code repositories. The training objective is simple: predict the next token. Wrong predictions trigger tiny weight adjustments via <strong>backpropagation</strong>. Every learnable parameter from the architecture section — the embedding matrix, the Q/K/V attention matrices, the feed-forward weights in every layer — is tuned through this process. Nothing is hand-programmed; the model discovers what to pay attention to and how to process meaning entirely from data.',
            'The same backpropagation algorithm popularized in 1986 (see Slide 4) now runs across thousands of GPUs simultaneously, allowing the model to learn from errors at an astronomical scale.',
            'This process takes months on thousands of GPUs and costs millions of dollars. The result? A <strong>base model</strong> that can complete sentences, generate code, and recall facts—but often produces rambling or unhelpful outputs.'
        ],
        bullets: [
            '<strong>Scaling Laws:</strong> Predictable relationship: more parameters + more data + more compute = better performance',
            'Training data cutoff means models may lack knowledge of events after training (the cutoff varies by model)',
            '<strong>Data Quality:</strong> Curation matters as much as scale—models trained on smaller but higher-quality data can outperform those trained on larger, noisier datasets',
            'Base models understand language structure but haven\'t learned to be helpful assistants yet'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>Analogy:</strong> Pre-training is like teaching a child to predict the next word in stories. They learn grammar, vocabulary, and facts—but not how to hold a conversation or follow instructions helpfully.'
        },
        resources: [
            { type: 'video', title: 'Neural Networks & Backprop', meta: '2+ hrs · Andrej Karpathy', url: 'https://www.youtube.com/watch?v=VMj-3S1tku0' },
            { type: 'video', title: 'What is Backpropagation?', meta: '14 min · 3Blue1Brown', url: 'https://www.3blue1brown.com/lessons/backpropagation' }
        ]
    },
    {
        category: 'train',
        badge: 'Training',
        title: '11. Post-Training',
        description: 'Post-training transforms a knowledgeable but unruly base model into a helpful, safe, and aligned assistant.',
        paragraphs: [
            'Base models know a lot but behave poorly—generating offensive content, refusing simple requests, or rambling endlessly. <strong>Post-training</strong> teaches them to be useful assistants through two key techniques:',
            '<strong>Supervised Fine-Tuning (SFT):</strong> Humans write ideal responses to thousands of prompts. The model learns to mimic this helpful behavior.',
            '<strong>Reinforcement Learning from Human Feedback (RLHF):</strong> Humans rank multiple model responses (A > B > C). A separate <strong>reward model</strong> is trained on these preferences to score new outputs. Then reinforcement learning optimizes the language model to produce responses that score highly. This two-step process works but is complex and can be unstable.',
            '<strong>Direct Preference Optimization (DPO):</strong> A modern alternative that skips the reward model entirely—it directly optimizes the language model to prefer better responses using the same human preference data. This makes training simpler, more stable, and more efficient.'
        ],
        bullets: [
            '<strong>DPO (Direct Preference Optimization):</strong> Optimizes preferences directly from preference data, without a separate reward model—simpler and often more stable than RLHF',
            'Post-training instills safety guardrails (refusing harmful requests)',
            'Models learn conversational norms: being concise, admitting uncertainty, citing sources',
            'Trade-off: alignment reduces raw creativity and capability slightly'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Safety vs. Capability:</strong> Post-training trades some raw capability for alignment. An aligned model might refuse edge-case requests a base model would attempt—prioritizing safety over unbounded helpfulness.'
        },
        resources: [
            { type: 'video', title: 'RLHF, Clearly Explained', meta: '18 min · StatQuest', url: 'https://www.youtube.com/watch?v=qPN_XZcJf_s' },
            { type: 'video', title: 'RLHF in 4 Minutes', meta: '4 min · Sebastian Raschka', url: 'https://www.youtube.com/watch?v=vJ4SsfmeQlk' }
        ]
    },
    {
        category: 'train',
        badge: 'Training',
        title: '12. Bias, Fairness & Limitations',
        description: 'AI models inherit the biases, gaps, and perspectives present in their training data—they are mirrors, not arbiters of truth.',
        paragraphs: [
            'Training data comes from the internet, books, and human-generated content—all of which contain biases, stereotypes, and uneven representation. Models learn these patterns just as they learn grammar and facts. If training data overrepresents certain demographics or perspectives, the model will too.',
            '<strong>Post-training alignment</strong> can reduce some harmful outputs (e.g., refusing to generate hate speech), but it doesn\'t eliminate underlying biases. A model might still generate biased resume summaries, make assumptions based on names, or reflect cultural stereotypes—even when trying to be helpful.',
            'This matters in high-stakes domains: healthcare (misdiagnosis patterns), hiring (resume screening bias), legal systems (risk assessment), education (unequal tutoring quality). No model is "objective"—all reflect their training data\'s worldview.'
        ],
        bullets: [
            '<strong>Sources of Bias:</strong> Training data imbalances, historical stereotypes, language and cultural gaps, majority perspectives dominating',
            '<strong>Types of Harm:</strong> Stereotyping, erasure (underrepresented groups), performance gaps (works better for some demographics)',
            '<strong>Mitigation Strategies:</strong> Diverse training data, red-teaming for harmful outputs, constitutional AI principles, ongoing monitoring',
            '<strong>User Responsibility:</strong> Critical evaluation of outputs, awareness of limitations, human oversight in high-stakes decisions'
        ],
        callout: {
            type: 'insight',
            content: '<strong>No Silver Bullet:</strong> Bias mitigation is an ongoing process, not a solved problem. Even the most carefully trained models can produce biased outputs. The goal is harm reduction and transparency, not perfection. Always apply human judgment, especially in consequential decisions.<br><br><em>Training is now complete and the model\'s weights are frozen. Next: what happens when you actually use it.</em>'
        },
        resources: [
            { type: 'video', title: 'AI Bias Explained', meta: '9 min · TEDx', url: 'https://www.youtube.com/watch?v=59bMh59JQDo' },
            { type: 'article', title: 'AI 600-1: AI Risk Management Framework', meta: 'NIST · Comprehensive framework', url: 'https://www.nist.gov/artificial-intelligence/executive-order-safe-secure-and-trustworthy-artificial-intelligence' }
        ]
    },
    {
        category: 'infer',
        badge: 'Inference',
        title: '13. The Frozen State & Prompt Stack',
        description: 'After training, model weights are frozen. Inference uses a structured prompt stack to generate responses without learning.',
        paragraphs: [
            'Once training completes, the model\'s weights are <strong>locked</strong>. Inference (generating responses) reads these weights but never modifies them. This is why chatting doesn\'t teach the model anything permanent—corrections only affect the current conversation\'s context.',
            'When you send a message, the system assembles a <strong>prompt stack</strong>: (1) <strong>System Prompt</strong> (hidden instructions defining persona), (2) <strong>Conversation History</strong> (prior messages re-sent on each turn), and (3) <strong>User Prompt</strong> (your message).',
            'This entire stack is tokenized and fed through the "skyscraper" of Transformer layers (the stacked architecture from Slide 8). The model then enters an <strong>Autoregressive Loop</strong>: it predicts one token, appends it to the prompt, and runs the entire process again to find the next token. This is why you see text appear word-by-word (streaming).'
        ],
        bullets: [
            '<strong>Frozen Weights:</strong> Feedback doesn\'t update the model\'s "brain" permanently',
            '<strong>Context Window:</strong> The fixed maximum number of tokens the model can "see" at once',
            '<strong>Prompt Assembly:</strong> System + History + User message = the full input',
            '<strong>Autoregressive Loop:</strong> The model generates one token at a time, feeding its own output back as input'
        ],
        callout: {
            type: 'note',
            content: '<strong>The Context Bottleneck:</strong> Because the entire history is re-processed every turn, long conversations get slower and more expensive. Once the context window is full, the model "forgets" the earliest parts of the chat.'
        },
        resources: [
            { type: 'article', title: 'Claude Prompting Best Practices', meta: 'Anthropic · Prompt engineering', url: 'https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices' },
            { type: 'video', title: 'Context Engineering Explained', meta: '17 min', url: 'https://www.youtube.com/watch?v=p6s82Ax8yrs' }
        ]
    },
    {
        category: 'infer',
        badge: 'Inference',
        title: '14. What Happens When You Send a Message?',
        description: 'A step-by-step walkthrough of the full inference pipeline—from keystrokes to streaming tokens.',
        paragraphs: [
            'When you press "Send," a precise sequence of operations begins. Understanding this pipeline demystifies why models behave the way they do—and where things can go wrong.',
            '<strong>Step 1 — Prompt Assembly:</strong> The system combines hidden instructions (system prompt), prior conversation turns (history), and your new message into a single text block. This assembled prompt is the model\'s <em>entire</em> view of the conversation.',
            '<strong>Step 2 — Tokenization:</strong> The full prompt is split into token IDs (Slide 2). A short question might become 20 tokens; a long conversation with system instructions might be 4,000+. Each token costs compute and counts against the context window.',
            '<strong>Step 3 — Embedding & Position:</strong> Each token ID is converted into a vector (Slide 3), and positional encodings are added so the model knows word order.',
            '<strong>Step 4 — The Transformer Forward Pass:</strong> The full sequence of vectors flows upward through every Transformer layer (the "skyscraper" from Slide 8). Attention lets tokens communicate; feed-forward networks refine meaning. After dozens of layers, the <em>final</em> token\'s vector emerges at the top, encoding the model\'s best prediction for what comes next.',
            '<strong>Step 5 — Token Selection:</strong> That final vector is compared against the entire vocabulary to produce scores (logits). A token is selected (details in the next slide) and streamed to you.',
            '<strong>Step 6 — The Loop:</strong> The selected token is appended to the sequence, and <em>the entire process repeats</em> from Step 3. This is the autoregressive loop — one token at a time until the model produces a stop token or hits the output limit.'
        ],
        bullets: [
            '<strong>Every Turn Re-Processes Everything:</strong> The full prompt (system + history + your message + tokens generated so far) is re-tokenized and re-run through all layers on every single token generation step',
            '<strong>Cost Scales with Length:</strong> Longer conversations = more tokens = more compute per generated token',
            '<strong>Why Streaming Exists:</strong> Each token is available as soon as it\'s generated, so you see text appear word-by-word rather than waiting for the full response',
            '<strong>The Context Window Limit:</strong> When the total token count (input + output) exceeds the context window, the model can no longer see the earliest parts of the conversation'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Key Insight:</strong> There is no "memory" between turns. Every time you send a message, the model sees the entire conversation from scratch—as if reading it for the first time. This is why the same question can get different answers in different conversations, and why long chats eventually "forget" early context.'
        },
        resources: [
            { type: 'video', title: 'How GPT Models Work', meta: '10 min · Visual walkthrough', url: 'https://www.youtube.com/watch?v=wjZofJX0v4M' },
            { type: 'article', title: 'What Is Inference?', meta: 'Anthropic · Fundamentals', url: 'https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices' }
        ]
    },
    {
        category: 'infer',
        badge: 'Inference',
        title: '15. The Selection Dice Roll',
        description: 'The final step: turning a refined vector back into a human word.',
        paragraphs: [
            'At the roof of the skyscraper, the model has a highly refined vector. It compares this "thought" against its entire vocabulary and gives every word a score (<strong>Logits</strong>).',
            'These scores are turned into probabilities. The model doesn\'t "know" the answer; it just knows that "Medici" has a 75% chance of being the next right word.',
            '<strong>Temperature</strong> controls the "risk." High temperature = roll the dice on lower-probability words (creativity). Low temperature = pick the most likely word (predictability).'
        ],
        bullets: [
            '<strong>Logits:</strong> Raw scores for every possible token in the vocabulary',
            '<strong>Temperature:</strong> A slider that adjusts how much "risk" the model takes in selection',
            '<strong>Probabilistic:</strong> The model is optimized for "plausibility," not necessarily truth',
            '<strong>Streaming:</strong> Tokens are sent to the user as they are generated, one by one'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Prediction, not Truth:</strong> If the most statistically likely next word is a hallucination, the model will pick it because its math told it to, not because it "wants" to lie. It is a statistical mirror, not a database.<br><br><em>That\'s the core pipeline—architecture, training, and inference. Next: advanced capabilities that extend this foundation.</em>'
        },
        resources: [
            { type: 'video', title: 'Why LLMs Hallucinate', meta: 'Practical · Video', url: 'https://www.youtube.com/watch?v=cfqtFvWOfg0' },
            { type: 'article', title: 'Why language models hallucinate', meta: 'OpenAI · Research', url: 'https://openai.com/index/why-language-models-hallucinate/' }
        ]
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: '16. RAG: Giving Models Access to Knowledge',
        description: 'RAG extends simple chat by letting models retrieve and use external documents—overcoming knowledge cutoffs and accessing private data.',
        paragraphs: [
            'Simple chat is limited to the model\'s training data (with a knowledge cutoff date) and has no access to your private documents. <strong>Retrieval-Augmented Generation (RAG)</strong> solves this by dynamically fetching relevant information and inserting it into the model\'s context before generating a response.',
            '<strong>How it works:</strong> (1) User asks a question. (2) System searches a document collection using <strong>semantic search</strong> (embeddings from the architecture section, stored in vector databases—specialized indexes optimized for finding similar vectors quickly). (3) Retrieved documents are injected into the prompt. (4) Model generates an answer informed by both its training and the retrieved text.',
            'RAG transforms models from "closed-book" systems (answering from memory) to "open-book" systems (consulting references). This dramatically reduces hallucinations on factual questions and enables models to work with proprietary data, internal wikis, legal databases, and real-time information.'
        ],
        bullets: [
            '<strong>Key Benefits:</strong> Reduces hallucinations, enables citations/sources, keeps knowledge current without retraining, works with private data',
            '<strong>Use Cases:</strong> Customer support (search help docs), legal research (case law), enterprise Q&A (internal wikis), academic research (paper collections)',
            '<strong>Infrastructure:</strong> Requires embedding model + vector database (Pinecone, Weaviate, Chroma) to enable fast semantic search',
            '<strong>Trade-offs:</strong> Adds latency, requires document indexing, quality depends on retrieval relevance'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>Analogy:</strong> RAG is like allowing a student to bring textbooks into an exam. They still use reasoning and comprehension—but can look up specific facts instead of guessing from memory. The better the textbooks (documents) and search skills (retrieval), the better the answers.'
        },
        resources: [
            { type: 'video', title: 'What is RAG?', meta: '6 min · IBM', url: 'https://youtube.com/watch?v=T-D1OfcDW1M' },
            { type: 'article', title: 'OpenAI Embeddings Guide', meta: 'Technical docs · Foundation for RAG', url: 'https://platform.openai.com/docs/guides/embeddings' },
            { type: 'video', title: 'Embeddings Explained', meta: '18 min · 3D visualizations', url: 'https://www.youtube.com/watch?v=eUbKYEC0D3Y' }
        ]
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: '17. Beyond Text: Multimodal Models & Tool Use',
        description: 'Models now extend beyond simple text chat by processing multiple input types and taking real-world actions.',
        paragraphs: [
            'Simple text chat is just the beginning. Modern AI systems extend in two complementary directions: <strong>richer inputs</strong> (multimodal) and <strong>actionable outputs</strong> (tool use).',
            '<strong>Multimodal Models — Processing More Than Text:</strong> Modern models can understand images, audio, and video alongside text. Images are split into patches and tokenized (like words), audio is converted to spectrograms, video combines both. All modalities are projected into a unified embedding space where a picture of a cat and the word "cat" occupy nearby points.',
            '<strong>Examples:</strong> <strong>GPT-4V</strong> and <strong>Claude</strong> (vision + text) let you upload diagrams, screenshots, or photos and ask questions. <strong>Gemini</strong> handles text + images + video. <strong>Whisper</strong> transcribes audio. <strong>DALL-E/Midjourney</strong> generate images from text descriptions.',
            '<strong>Tool Use (Function Calling) — Taking Actions:</strong> LLMs are prediction engines, not calculators or databases. Tool use lets models request external actions: run calculations, search the web, query databases, execute code, send emails, control APIs. The model outputs a structured request like <code>{"tool": "calculator", "input": "sqrt(144)"}</code>, your system executes it, and the result feeds back into the conversation.',
            '<strong>Examples:</strong> Web search for current information, code interpreters for data analysis, API calls for booking flights, database queries for customer records. Modern models can chain multiple tools sequentially to accomplish complex tasks.'
        ],
        bullets: [
            '<strong>Multimodal Use Cases:</strong> Analyze charts/diagrams, describe images for accessibility, transcribe meetings, generate creative visuals, ask questions about photos',
            '<strong>Tool Use Cases:</strong> Precise calculations, real-time data (weather, stock prices), code execution, database access, workflow automation',
            '<strong>Why It Matters:</strong> Together, these capabilities transform models from "text conversationalists" into versatile systems that can perceive richer inputs and take concrete actions',
            '<strong>Technical Note:</strong> The same Transformer architecture handles all modalities—only the tokenization step differs, which is why capabilities expanded rapidly'
        ],
        callout: {
            type: 'insight',
            content: '<strong>The Extension Pattern:</strong> Multimodal extends what models can <em>perceive</em> (text → text + images + audio + video). Tool use extends what models can <em>do</em> (generate text → generate text + execute actions). Combined, they create systems that interact with the world more like humans do.'
        },
        resources: [
            { type: 'video', title: 'How Multimodal Models Work', meta: '12 min · Visual explanation', url: 'https://www.youtube.com/watch?v=vAmKB7iPkWw' },
            { type: 'video', title: 'Function Calling Tutorial', meta: 'OpenAI · Tool use', url: 'https://www.youtube.com/watch?v=4Dj3j6WqcG0' },
            { type: 'article', title: 'Claude Vision Capabilities', meta: 'Anthropic · Multimodal', url: 'https://docs.anthropic.com/en/docs/build-with-claude/vision' }
        ]
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: '18. Reasoning: Two Paradigms',
        description: 'Reasoning capability evolved from simple prompting tricks to fundamental architectural changes in how models think.',
        paragraphs: [
            'Early language models struggled with multi-step reasoning—jumping to conclusions or making arithmetic errors. Two approaches emerged, representing different philosophies about where "thinking" should happen.',
            '<strong>Chain-of-Thought (CoT) — 2022:</strong> A <strong>prompting technique</strong> where you ask the model to "think step by step." By generating intermediate reasoning steps, models dramatically improve on math, logic, and complex questions—but the reasoning is visible, uses more output tokens (since reasoning steps appear in the response), and requires user-side prompt engineering.',
            '<strong>Inference-Time Compute (2024–2025):</strong> Reasoning <strong>built into the model itself</strong>. Instead of immediately answering, the model generates hidden reasoning tokens—internal "thoughts" not shown to the user—exploring multiple paths and backtracking from errors. Models like o1 and DeepSeek-R1 can "think" for seconds or minutes before responding. This shifts responsibility from <strong>prompt engineering</strong> (user) to <strong>model architecture</strong> (system): reasoning happens automatically when problems are hard.'
        ],
        bullets: [
            '<strong>CoT (2022):</strong> Visible reasoning, user controls format, uses more output tokens, works with any model',
            '<strong>Inference-Time Compute (2024+):</strong> Hidden reasoning, model controls strategy, compact visible output (but high total compute), requires specialized training',
            '<strong>Performance Trade-off:</strong> Reasoning models are slower and more expensive per query, but dramatically more accurate on complex tasks',
            '<strong>Scaling Law:</strong> Test-time compute scales inference—more thinking time → better results (unlike training compute, which scales the model)'
        ],
        callout: {
            type: 'insight',
            content: '<strong>The Evolution:</strong> CoT proved that models <em>can</em> reason if given space to think. Inference-time compute took the next step: teaching models <em>when and how</em> to allocate that thinking automatically. This mirrors human cognition: <strong>System 1</strong> (fast, intuitive, "autopilot") vs. <strong>System 2</strong> (slow, deliberate, "effortful thinking"). Reasoning models represent AI\'s first true "pause and think" capability.'
        },
        resources: [
            { type: 'video', title: 'Chain-of-Thought Explained', meta: '8 min · 2022 breakthrough', url: 'https://www.youtube.com/watch?v=AFE6x81AP4k' },
            { type: 'video', title: 'Test-Time Scaling', meta: '12 min · Inference-time reasoning', url: 'https://www.youtube.com/watch?v=NbE8MoR8mPw' },
            { type: 'article', title: 'OpenAI o1 System Card', meta: 'Technical report · Hidden reasoning', url: 'https://openai.com/index/openai-o1-system-card/' }
        ]
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: '19. Agentic Workflows',
        description: 'Combining reasoning, tools, and planning creates autonomous agents that can accomplish complex multi-step tasks.',
        paragraphs: [
            'An <strong>agent</strong> is fundamentally different from simple LLM chat. While a standard chatbot generates one response and stops, an agent operates in a <strong>perception-reasoning-action loop</strong>: it observes the current state, reasons about what to do next, takes actions using tools, observes the results, and continues until the goal is achieved.',
            '<strong>The ReAct Pattern (Reasoning + Acting):</strong> The core framework is simple but powerful: <strong>Thought → Action → Observation → Repeat</strong>. Example: debugging failing code. <em>Thought:</em> "The test fails with ImportError—need to check what\'s imported." <em>Action:</em> Read the file. <em>Observation:</em> "Missing \'requests\' import." <em>Thought:</em> "Add the import." <em>Action:</em> Edit file. <em>Observation:</em> "Test now passes." The model iteratively adjusts its plan based on real feedback.',
            '<strong>Production Systems:</strong> Modern frameworks like <strong>LangGraph</strong>, <strong>AutoGPT</strong>, and <strong>CrewAI</strong> power coding assistants (Cursor, Devin), research tools, and customer support bots. Agents excel at <strong>planning</strong> (task decomposition), <strong>memory</strong> (maintaining state), and <strong>reflection</strong> (self-verification). However, they\'re expensive (dozens of LLM calls per task), can be unreliable (stuck in loops, wrong tool choices), and need monitoring. They work best for well-defined tasks with clear success criteria.'
        ],
        bullets: [
            '<strong>Agent vs. Chat:</strong> Agents loop autonomously (perception-action-feedback) instead of single-turn responses',
            '<strong>Real-World Use Cases:</strong> Coding assistants (Cursor, Devin), research automation (paper search + summarization), customer support (knowledge base search + ticket creation), data analysis (load + clean + visualize)',
            '<strong>Core Components:</strong> Planning (task decomposition), Memory (state persistence), Reflection (self-verification), Tool orchestration (APIs, code execution, databases)',
            '<strong>Challenges:</strong> Cost (multiple LLM calls), reliability (can fail or loop indefinitely), monitoring (need human oversight for high-stakes tasks)',
            '<strong>Best Practices:</strong> Clear success criteria, robust error handling, human-in-the-loop for critical decisions, cost limits to prevent runaway loops'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>Analogy:</strong> A standard LLM is a smart person answering questions. An agent is that person given a desk, computer, calculator, notepad, internet access, and the instruction to "figure it out"—they can try things, see what works, and iterate until the job is done.'
        },
        resources: [
            { type: 'video', title: 'What are AI Agents?', meta: '12 min · IBM Technology', url: 'https://www.youtube.com/watch?v=F8NKVhkZZWI' },
            { type: 'article', title: 'LangGraph: Agent Framework', meta: 'LangChain · Build production agents', url: 'https://langchain-ai.github.io/langgraph/' }
        ]
    },
    {
        category: 'adv',
        badge: 'Looking Ahead',
        title: '20. Looking Ahead',
        description: 'A look beyond the core content: emerging architectures and research directions.',
        paragraphs: [
            'Transformers dominate today, but several directions are already in production or heavy research.',
            '<strong>JEPA (Joint Embedding Predictive Architecture):</strong> Yann LeCun\'s vision—predict in <em>representation space</em>, not pixels or tokens. Goal: sample-efficient learning, world models, planning. V-JEPA and I-JEPA are early implementations.',
            '<strong>Mamba & State Space Models (SSMs):</strong> Linear or near-linear sequence complexity instead of quadratic attention. Recurrent state, long context without the same memory cost. Used in some long-context and efficient LLMs.',
            '<strong>Mixture of Experts (MoE):</strong> Sparse activation—route each token to a subset of "expert" sub-networks instead of one dense stack. Lets you scale total parameters (e.g. 400B+) while keeping compute per token similar. Used in DeepSeek-V3, Llama-4, Gemini-2.5, Mixtral.',
            '<strong>RWKV & RetNet:</strong> RNN-like inference (O(1) memory) with parallelizable training. "Successor to Transformer" narrative; constant-memory decoding and long context. RWKV-7 and RetNet are in active use.',
            '<strong>Hybrids:</strong> Models that mix attention with SSMs and/or MoE (e.g. Jamba: attention + Mamba + MoE; upcoming Qwen hybrids, linear attention). "Attention was never enough"—combining mechanisms is a major trend.'
        ],
        bullets: [
            '<strong>JEPA:</strong> Predict in latent space → world models, planning',
            '<strong>Mamba/SSMs:</strong> Linear-time sequences → long context, efficiency',
            '<strong>MoE:</strong> Sparse experts → scale parameters without scaling compute per token',
            '<strong>RWKV/RetNet:</strong> Recurrent inference, parallel training → O(1) decode',
            '<strong>Hybrids:</strong> Attention + SSM + MoE in one model; already in production'
        ],
        resources: [
            { type: 'video', title: 'JEPA — A Path Towards Autonomous Machine Intelligence', meta: 'Paper Explained · LeCun', url: 'https://www.youtube.com/watch?v=jSdHmImyUjk' },
            { type: 'video', title: 'Mamba: Linear-Time Sequence Modeling (Paper Explained)', meta: 'Selective State Spaces', url: 'https://www.youtube.com/watch?v=9dSkvxS2EB0' },
            { type: 'video', title: 'Intuition behind Mamba and State Space Models', meta: 'Visual · SSMs', url: 'https://www.youtube.com/watch?v=BDTVVlUU1Ck' },
            { type: 'video', title: 'What is Mixture of Experts?', meta: '7 min · IBM', url: 'https://www.youtube.com/watch?v=sYDlVVyJYn4' },
            { type: 'video', title: 'RWKV: Reinventing RNNs for the Transformer Era', meta: 'Paper Explained', url: 'https://www.youtube.com/watch?v=x8pW19wKfXQ' },
            { type: 'video', title: 'Retentive Network: A Successor to Transformer', meta: 'Paper Explained', url: 'https://www.youtube.com/watch?v=ec56a8wmfRk' }
        ]
    },
    {
        category: 'adv',
        badge: 'Conclusion',
        title: '21. Recap: Key Learnings',
        description: 'Understanding how AI models work transforms you from a passive user into an informed practitioner.',
        paragraphs: [
            'Modern AI isn\'t magic. It\'s a <strong>high-fidelity statistical mirror</strong> of human-created text, trained on trillions of tokens to predict plausible continuations. Understanding this—the frozen weights, tokenization quirks, training phases, and inference mechanics—transforms AI from a mysterious black box into a predictable, engineered system whose failures and capabilities make sense.',
            '<strong>Key Mental Models to Internalize:</strong> (1) <strong>Frozen Artifact</strong>—inference never teaches the model; corrections only affect the current context. (2) <strong>Plausibility Over Truth</strong>—the model outputs what statistically "sounds right," not what is factually correct. (3) <strong>Token-Level Reasoning</strong>—models don\'t see letters or words as humans do; tokenization explains why they struggle with spelling, counting characters, or reversing text. (4) <strong>Context is Everything</strong>—the prompt stack (system + history + user input) is the model\'s entire universe; once context fills up, early information vanishes.',
            '<strong>Practical Decision Framework:</strong> When should you use different techniques? <strong>Prompt engineering</strong> (simple, cheap, instant, but limited)—start here for most tasks. <strong>RAG</strong> (retrieval-augmented generation)—when you need factual accuracy, citations, or access to private/current data. <strong>Fine-tuning</strong> (expensive, permanent, requires ML expertise)—when you need specialized behavior or domain expertise that can\'t fit in context; continues training on domain-specific data, permanently adjusting weights for specialized behavior like medical diagnosis or legal analysis. <strong>Reasoning models</strong> (o1, DeepSeek-R1)—when accuracy matters more than speed/cost, especially for math, code, and logic-heavy tasks.'
        ],
        bullets: [
            '<strong>Core Understanding:</strong> Models are frozen prediction engines, not databases or calculators—they compress training data into patterns, not facts',
            '<strong>Strengths:</strong> Language fluency, pattern recognition, creative generation, code completion, summarization, explaining concepts',
            '<strong>Weaknesses:</strong> Precise arithmetic (use calculators), recent events (use RAG or web search), verifying facts (requires external validation), counting/spelling (tokenization limitations)',
            '<strong>When to Choose What:</strong> Prompt engineering (default), RAG (factual accuracy/citations), fine-tuning (specialized domains), reasoning models (complex logic/math)',
            '<strong>The Empowered User:</strong> Architecture knowledge lets you predict failures, debug issues, choose appropriate tools, and use AI as an informed practitioner rather than a passive consumer'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Final Thought:</strong> The "magic" of AI isn\'t that it thinks—it\'s that billions of mathematical operations, carefully arranged and trained on trillions of tokens, compress human knowledge into a queryable, frozen artifact. You now understand the machine: tokenization → embeddings → attention → prediction. This knowledge transforms you from a user who hopes AI works into a practitioner who knows <em>why</em> it works—and when it won\'t.'
        },
        resources: [
            { type: 'video', title: 'Intro to Large Language Models', meta: '1 hour · Andrej Karpathy', url: 'https://www.youtube.com/watch?v=zjkBMFhNj_g' }
        ]
    }
];