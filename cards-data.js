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
            content: '<strong>Training vs. Inference:</strong> Training updates the model\'s "brain" (weights)—a massive structure that can contain trillions of parameters. Inference is the act of querying that "frozen" brain. Chatting provides temporary context, but it does not permanently teach the model or update its knowledge base.'
        },
        resources: [
            { type: 'video', title: 'Generative AI in a Nutshell', meta: '18 min · Henrik Knibbe', url: 'https://www.youtube.com/watch?v=2IK3DFHRFfw' },
            { type: 'interactive', title: 'OKAI — Interactive Intro to AI', meta: 'Brown University', url: 'https://okai.brown.edu/' },
            { type: 'video', title: 'Large Language Models, briefly', meta: '8 min · 3Blue1Brown', url: 'https://www.youtube.com/watch?v=LPZh9BOjkQs' }
        ]
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
            '<strong>Tokens are pieces, not words:</strong> The model often sees subwords like <code>un</code> + <code>believ</code> + <code>able</code>'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Key idea:</strong> Most language models are trained on one core objective: <em>predict the next token</em>. That single skill can look like reasoning, writing, or coding—but it\'s still prediction, not guaranteed "truth" or perfect calculation.'
        },
        resources: [
            { type: 'tool', title: 'OpenAI Tokenizer', meta: 'Interactive · Try BPE', url: 'https://platform.openai.com/tokenizer' },
            { type: 'video', title: 'Byte Pair Encoding Explained', meta: '7 min · Tokenization', url: 'https://www.youtube.com/watch?v=4A_nfXyBD08' },
            { type: 'video', title: 'Build GPT Tokenizer', meta: '2h 13min · Andrej Karpathy', url: 'https://www.youtube.com/watch?v=zduSFxRajkE' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '3. From Numbers to Meaning: Embeddings & Latent Space',
        description: 'Token IDs are just numbers—embeddings transform them into rich, meaningful representations in a high-dimensional "Latent Space."',
        paragraphs: [
            'After tokenization, the model has a sequence of token IDs—integers like [4829, 2121, 8945]. But numbers alone are meaningless. The model needs to understand what each token <em>represents</em>.',
            'Enter <strong>embeddings</strong>: the model looks up each token ID in a massive learned table (the <strong>embedding matrix</strong>) and retrieves its corresponding <strong>vector</strong>—a list of hundreds or thousands of numbers. Each dimension captures some aspect of meaning: semantic properties, grammatical role, contextual patterns learned during training.',
            'These vectors live in <strong>Latent Space</strong>, a high-dimensional coordinate system where similar meanings cluster together. Early AI tried to pre-define these features (e.g., "Gender" or "Size"), but modern models discover them organically. A vector might have 4,096 dimensions, each representing a "latent" feature the model found useful, even if humans don\'t have a name for it.',
            'The model also adds <strong>positional encodings</strong>—patterns that tell it where each token appears in the sequence. This step happens <em>after</em> the embedding lookup and <em>before</em> the first Transformer layer; it gives self-attention access to position so it can weigh tokens by both content and location. These encodings can be learned during training or use fixed mathematical patterns (like sine waves at different frequencies) to represent position. Without position, "dog bites man" and "man bites dog" would look identical. Position + meaning = the full input representation that flows into the Transformer layers.'
        ],
        bullets: [
            '<strong>Embedding Lookup:</strong> Token ID → retrieve learned vector from embedding table',
            '<strong>Latent Space:</strong> An organic "map of meaning" where distance = difference in concept',
            '<strong>Learned Features:</strong> Models discover their own N-dimensional features (e.g., "blue," "tall," "hairy") during training',
            '<strong>Positional Encoding:</strong> Adds order information so word sequence matters; applied after embed, before first layer—enables self-attention to use position',
            '<strong>Fixed Structure:</strong> The prompt is a tensor of shape (sequence length × dimension). Through every layer this shape stays the same; only the vector values change (attention and FFNs update the numbers, not the dimensions or token count)'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>The Flavor Profile Analogy:</strong> If tokenization assigns each word a locker number, embeddings are the contents—a profile describing the word\'s "flavor." One dimension might be "sweetness," another "spiciness." In a 4,096-dimensional space, the model creates a hyper-detailed profile for every concept.'
        },
        resources: [
            { type: 'video', title: 'Tokens and Embeddings', meta: '7 min · Visual', url: 'https://www.youtube.com/watch?v=izbifbq3-eI' },
            { type: 'video', title: 'Language Models & Transformers', meta: '20 min · Computerphile', url: 'https://www.youtube.com/watch?v=rURRYI66E54' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '4. Feed-Forward Neural Networks (FFNN)',
        description: 'The "classic" neural network architecture where data flows in one direction—the ancestor of all modern AI.',
        paragraphs: [
            '<strong>Feed-forward neural networks</strong> were the first successful neural network architecture, dating back to the <strong>Perceptron</strong> (1958). These networks process information in a single direction: input → hidden layers → output, with no loops or memory.',
            'Early perceptrons could only solve simple linear problems. The breakthrough came with <strong>multilayer networks</strong> and <strong>backpropagation</strong> (1986), which allowed models to learn complex, non-linear relationships by adjusting weights across multiple layers.',
            'While these "classic" FFNNs are the foundation of deep learning, they have a fundamental limitation: they process each input independently. They have no "memory" of what came before, making them unsuitable for sequential data like sentences or music on their own.',
            '<strong>Note on Terminology:</strong> In modern Transformers, we still use small FFNNs <em>inside</em> every layer (see Slide 10) to process tokens, but the overall Transformer architecture uses Attention to solve the "no memory" problem.'
        ],
        bullets: [
            '<strong>One-Way Flow:</strong> Data flows forward only (input → output), no feedback loops',
            '<strong>No Memory:</strong> Each input processed independently, no context from previous inputs',
            '<strong>Backpropagation (1986):</strong> Enables training of deep multilayer networks',
            '<strong>Limitation:</strong> Cannot model temporal dependencies or sequential patterns'
        ],
        callout: {
            type: 'insight',
            content: '<strong>The Foundation:</strong> Feed-forward networks established the basic architecture and training methods (backpropagation) that all subsequent neural networks build upon. Their limitation—no memory—led to the development of RNNs for sequential data.'
        },
        resources: [
            { type: 'video', title: 'The Essential Main Ideas of Neural Networks', meta: '19 min · StatQuest', url: 'https://youtu.be/CqOfi41LfDw?si=vGamzRxa1mtcQ3nf' },
            { type: 'video', title: 'What is Backpropagation?', meta: '14 min · 3Blue1Brown', url: 'https://www.3blue1brown.com/lessons/backpropagation' }
        ]
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
            { type: 'video', title: 'Recurrent Neural Networks (RNN) - Clearly Explained', meta: '16 min · StatQuest', url: 'https://www.youtube.com/watch?v=AsNTP8Kwu80' }
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
            { type: 'video', title: 'LSTM Networks - Clearly Explained', meta: '20 min · StatQuest', url: 'https://www.youtube.com/watch?v=YCzL96nL7j0' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '7. Encoder-Decoder Architecture (Seq2Seq)',
        description: 'The "Seq2Seq" breakthrough introduced the "Thought Vector"—a way to turn a whole sentence into a single mathematical point.',
        paragraphs: [
            'RNNs and LSTMs process sequences, but they struggle with a key challenge: how do you transform one sequence into another sequence of different length? <strong>Encoder-decoder architecture</strong> (also called <strong>Seq2Seq</strong>) was invented in 2014 by Ilya Sutskever and Kyunghyun Cho to solve this.',
            'The architecture splits the task between two RNNs: The <strong>encoder</strong> processes the entire input sequence, compressing it into a single fixed-size vector in Latent Space called the <strong>context vector</strong> (often called a "Thought Vector"). The <strong>decoder</strong> then takes this single vector and "unpacks" it into a new sequence. (Note: "Thought Vector" was popular in early Seq2Seq work; modern papers typically just say "context vector.")',
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
        title: '8. Attention Mechanism',
        description: 'Attention allows models to focus on relevant parts of the input sequence, solving long-range dependency problems.',
        paragraphs: [
            'The <strong>attention mechanism</strong> was introduced to solve the long-range dependency problem. Instead of compressing all information into a single hidden state, attention allows the model to directly look at and weight relevant parts of the input sequence.',
            '<strong>Seq2Seq models</strong> (encoder-decoder architectures) first used attention for machine translation. The encoder processes the input sequence, and the decoder uses attention to focus on relevant encoder states when generating each output token. This was a breakthrough—attention helped models handle longer sequences and improved translation quality.',
            'However, these early attention-based models still used RNNs as their backbone. The encoder and decoder were still sequential RNNs, with attention layered on top. Attention solved the memory problem, but sequential processing remained the bottleneck.',
            'The key insight: if attention can solve dependencies, <strong>why do we need RNNs at all?</strong> This question led to the Transformer architecture.'
        ],
        bullets: [
            '<strong>Direct Access:</strong> Model can look at any part of the input sequence',
            '<strong>Weighted Focus:</strong> Attention scores determine which parts are most relevant',
            '<strong>Long-Range Dependencies:</strong> Solves the vanishing gradient problem',
            '<strong>Still Sequential:</strong> Early attention models used RNN backbones, limiting parallelization'
        ],
        callout: {
            type: 'insight',
            content: '<strong>The Key Question:</strong> If attention can solve long-range dependencies, why do we need sequential RNNs? This insight led to the Transformer: attention-only architecture that processes all tokens in parallel.'
        },
        resources: [
            { type: 'video', title: 'I Visualised Attention in Transformers', meta: '13 min · Gal Lahat', url: 'https://www.youtube.com/watch?v=RNF0FvRjGZk' },
            { type: 'article', title: 'Attention? Attention!', meta: '21 min · Lilian Weng', url: 'https://lilianweng.github.io/posts/2018-06-24-attention/' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '9. The Transformer Architecture',
        description: 'The Transformer is a stack of repeating layers that progressively refine token representations through attention and processing.',
        paragraphs: [
            'A Transformer isn\'t a single operation—it\'s a <strong>stack of identical layers</strong> (typically 12, 24, 48, or more) that process embeddings sequentially. Think of it as a skyscraper: each floor performs the same two operations on every token\'s vector as it passes through.',
            '<strong>Each Transformer layer contains:</strong> (1) A <strong>self-attention mechanism</strong> that lets tokens "communicate" and update their representations based on context, and (2) A <strong>feed-forward network</strong> that processes each token\'s vector independently, refining its meaning.',
            'As vectors flow upward through the stack, they accumulate increasingly abstract and context-aware information. Early layers capture basic patterns like grammar and syntax. Middle layers learn relationships and simple logic. Deep layers encode complex reasoning, nuanced meaning, and task-specific behavior.',
            'The power of Transformers comes from this <strong>deep, repeated processing</strong>. Each layer adds a small refinement, but stacking dozens of them allows the model to build sophisticated representations from simple token embeddings.',
            'Most modern LLMs (GPT, Claude, etc.) use only the decoder part of this architecture—<strong>decoder-only Transformers</strong>—without a separate encoder.'
        ],
        bullets: [
            '<strong>Layer Structure:</strong> Self-attention + feed-forward network, repeated N times',
            '<strong>Progressive Refinement:</strong> Each layer adds context and abstraction to token vectors',
            '<strong>Residual Connections:</strong> Original information is preserved as it flows upward, preventing information loss',
            '<strong>Depth = Capability:</strong> More layers enable more complex reasoning and pattern recognition'
        ],
        callout: {
            type: 'insight',
            content: '<strong>The Skyscraper Analogy:</strong> Ground floor tokens know only their own meaning. As they ride the elevator through dozens of floors—each adding context from surrounding words—they emerge at the top with rich, nuanced understanding of their role in the specific sentence.'
        },
        resources: [
            { type: 'video', title: 'Transformers, the tech behind LLMs', meta: '58 min · 3Blue1Brown', url: 'https://www.youtube.com/watch?v=KJtZARuO3JY' },
            { type: 'article', title: 'The Illustrated GPT-2', meta: 'Jay Alammar · Visualizing GPT-2', url: 'https://jalammar.github.io/illustrated-gpt2/' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '10. Inside a Transformer Layer',
        description: 'Each layer contains attention, feed-forward processing, normalization, and residual connections.',
        paragraphs: [
            'Each Transformer layer has a consistent structure that repeats throughout the network. Understanding this pattern reveals how information flows and accumulates:',
            '<strong>Self-Attention:</strong> Tokens "talk" to each other, updating their vectors based on the entire sequence context (details in the next slide).',
            '<strong>Feed-Forward Network:</strong> Each token\'s vector passes independently through a small neural network (expand to 4x size, transform, compress back). This is where the model "thinks" about each token individually after the attention step has gathered context.',
            '<strong>Terminology Note:</strong> While the <em>overall</em> Transformer isn\'t a feed-forward network (because of Attention), it uses these small FFNN blocks inside every layer as "processing stations" for the tokens.',
            '<strong>Layer Normalization:</strong> Before attention and before feed-forward, vectors are normalized to consistent scale. This prevents training instability in deep networks.',
            '<strong>Residual Connections:</strong> After attention and feed-forward, the original input is added back. These "skip connections" preserve information and enable training of 100+ layer networks.'
        ],
        bullets: [
            '<strong>Pattern:</strong> (Normalize → Attention → Add) → (Normalize → Feed-forward → Add)',
            '<strong>Feed-Forward:</strong> Expands vectors 4x, transforms, compresses back',
            '<strong>Residuals:</strong> Original information bypasses transformations, flows directly upward',
            '<strong>Key Insight:</strong> Without residuals and normalization, deep networks fail to train'
        ],
        callout: {
            type: 'note',
            content: '<strong>Engineering Breakthroughs:</strong> Residuals and normalization aren\'t just optimizations—they\'re what made Transformers scalable. These techniques enabled the jump from 12-layer to 96-layer models.'
        },
        resources: [
            { type: 'video', title: 'Layer Normalization Explained', meta: '8 min · Visual', url: 'https://www.youtube.com/watch?v=2V3Ud-FnvUs' },
            { type: 'video', title: 'Feedforward Neural Network Basics', meta: '5 min · Natalie Parde', url: 'https://www.youtube.com/watch?v=QK7GJZ94qPw' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '11. Attention: Query, Key, Value',
        description: 'Attention lets tokens look at each other through Query, Key, and Value vectors.',
        paragraphs: [
            'Attention solves a core problem: how can every token simultaneously understand context from all other tokens? The answer: Query, Key, Value.',
            'For each token, the model creates three vectors: <strong>Query (Q)</strong> — "What am I looking for?", <strong>Key (K)</strong> — "What do I represent?", and <strong>Value (V)</strong> — "What do I carry?"',
            'Each token\'s Query is compared (dot product) to all Keys, producing <strong>attention scores</strong>. High scores = relevance. These scores are normalized by <strong>softmax</strong>—a function that converts raw scores into probabilities that sum to 1. High scores get most of the "probability mass," while low scores get nearly zero. These weights are then used to compute a weighted average of all Values.',
            '<strong>Example:</strong> In "The bank by the river," "bank" compares its Query to all Keys. "River" scores high, so "bank" pulls in its Value, morphing toward "riverbank" not "financial institution."'
        ],
        bullets: [
            '<strong>Q, K, V:</strong> Three learned transformations of each embedding',
            '<strong>Scores:</strong> Query · Key (dot product), normalized by softmax',
            '<strong>Output:</strong> Weighted sum of Values',
            '<strong>Parallel:</strong> All tokens compute simultaneously'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>Library Search:</strong> Your Query is your question. Each book\'s Key is its description. High-scoring books contribute their content (Values). You get a weighted mix of relevant sources.'
        },
        resources: [
            { type: 'video', title: 'Attention in Transformers', meta: '26 min · 3Blue1Brown', url: 'https://www.youtube.com/watch?v=eMlx5fFNoYc' },
            { type: 'video', title: 'Illustrated Guide to Transformers Neural Network: A step by step explanation', meta: '15 min', url: 'https://www.youtube.com/watch?v=4Bdc55j80l8' },
            { type: 'article', title: 'Attention? Attention!', meta: '21 min · Lilian Weng', url: 'https://lilianweng.github.io/posts/2018-06-24-attention/' }
        ]
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '12. Multi-Head Attention',
        description: 'Models run attention multiple times in parallel, each "head" learning to focus on different relationships.',
        paragraphs: [
            'Models don\'t run attention once—they run it multiple times in parallel, called <strong>multi-head attention</strong>. A model might have 8, 16, or 32 attention heads operating simultaneously.',
            '<strong>Why Multiple Heads?</strong> Different heads learn to focus on different relationships. One head might specialize in syntax (subject-verb agreement), another in semantics (related concepts), another in coreference (pronouns to nouns). This gives the model multiple simultaneous "perspectives" on the sequence.',
            'Each head has its own Query, Key, and Value transformation matrices. They all run in parallel, producing separate attention outputs. These outputs are concatenated together and mixed through a final learned transformation.',
            '<strong>Self-Attention vs. Cross-Attention:</strong> Self-attention means tokens attend to other tokens in the same sequence—the input attends to itself. Cross-attention (used in encoder-decoder architectures) lets one sequence attend to a different sequence, like when translating from English to French.'
        ],
        bullets: [
            '<strong>Multiple Heads:</strong> 8-32 parallel attention operations with independent Q, K, V matrices',
            '<strong>Specialization:</strong> Each head learns different patterns (syntax, semantics, position)',
            '<strong>Combine:</strong> Concatenate all head outputs and mix through learned transformation',
            '<strong>Self-Attention:</strong> Tokens attend to the same sequence (most common in LLMs)'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>The Expert Panel:</strong> Instead of one judge evaluating relationships, you have a panel of 8-16 experts. Each expert focuses on different aspects—one on grammar, one on meaning, one on context. Their combined insights create a richer understanding than any single perspective.'
        },
        resources: [
            { type: 'video', title: 'Attention Is All You Need (walkthrough)', meta: '15 min · Visual', url: 'https://www.youtube.com/watch?v=wjZofJX0v4M' },
            { type: 'article', title: 'The Illustrated Transformer', meta: 'Jay Alammar · Multi-head', url: 'https://jalammar.github.io/illustrated-transformer/' }
        ]
    },
    {
        category: 'train',
        badge: 'Training',
        title: '13. Pre-Training',
        description: 'Pre-training is where models learn the patterns, facts, and structures of human knowledge from massive text datasets.',
        paragraphs: [
            'During pre-training, the model consumes <strong>trillions of tokens</strong> from books, websites, research papers, and code repositories. The training objective is simple: predict the next token. Wrong predictions trigger tiny weight adjustments via <strong>backpropagation</strong>.',
            'The same backpropagation algorithm popularized in 1986 (see Slide 4) now runs across thousands of GPUs simultaneously, allowing the model to learn from errors at an astronomical scale.',
            'This process takes months on thousands of GPUs and costs millions of dollars. The result? A <strong>base model</strong> that can complete sentences, generate code, and recall facts—but often produces rambling or unhelpful outputs.'
        ],
        bullets: [
            '<strong>Scaling Laws:</strong> Predictable relationship: more parameters + more data + more compute = better performance',
            'Training data cutoff means models may lack knowledge of events after training (the cutoff varies by model)',
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
        title: '14. Post-Training',
        description: 'Post-training transforms a knowledgeable but unruly base model into a helpful, safe, and aligned assistant.',
        paragraphs: [
            'Base models know a lot but behave poorly—generating offensive content, refusing simple requests, or rambling endlessly. <strong>Post-training</strong> teaches them to be useful assistants through two key techniques:',
            '<strong>Supervised Fine-Tuning (SFT):</strong> Humans write ideal responses to thousands of prompts. The model learns to mimic this helpful behavior.',
            '<strong>Reinforcement Learning from Human Feedback (RLHF):</strong> Humans rank multiple model responses (A > B > C). The model learns to maximize preference scores. Modern alternatives like <strong>Direct Preference Optimization (DPO)</strong> streamline this process—DPO skips the reward model entirely by directly optimizing the policy to prefer better responses, making training more stable and efficient.'
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
        title: '15. Bias, Fairness & Limitations',
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
            content: '<strong>No Silver Bullet:</strong> Bias mitigation is an ongoing process, not a solved problem. Even the most carefully trained models can produce biased outputs. The goal is harm reduction and transparency, not perfection. Always apply human judgment, especially in consequential decisions.'
        },
        resources: [
            { type: 'video', title: 'AI Bias Explained', meta: '9 min · TEDx', url: 'https://www.youtube.com/watch?v=59bMh59JQDo' }
        ]
    },
    {
        category: 'infer',
        badge: 'Inference',
        title: '16. The Frozen State & Prompt Stack',
        description: 'After training, model weights are frozen. Inference uses a structured prompt stack to generate responses without learning.',
        paragraphs: [
            'Once training completes, the model\'s weights are <strong>locked</strong>. Inference (generating responses) reads these weights but never modifies them. This is why chatting doesn\'t teach the model anything permanent—corrections only affect the current conversation\'s context.',
            'When you send a message, the system assembles a <strong>prompt stack</strong>: (1) <strong>System Prompt</strong> (hidden instructions defining persona), (2) <strong>Conversation History</strong> (prior messages re-sent on each turn), and (3) <strong>User Prompt</strong> (your message).',
            'This entire stack is tokenized and fed through the "skyscraper" of Transformer layers. The model then enters an <strong>Autoregressive Loop</strong>: it predicts one token, appends it to the prompt, and runs the entire process again to find the next token. This is why you see text appear word-by-word (streaming).'
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
        title: '17. The Selection Dice Roll',
        description: 'The final step: turning a "massaged" vector back into a human word.',
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
            content: '<strong>Prediction, not Truth:</strong> If the most statistically likely next word is a hallucination, the model will pick it because its math told it to, not because it "wants" to lie. It is a statistical mirror, not a database.'
        },
        resources: [
            { type: 'video', title: 'Why LLMs Hallucinate', meta: 'Practical · Video', url: 'https://www.youtube.com/watch?v=cfqtFvWOfg0' },
            { type: 'article', title: 'Why language models hallucinate', meta: 'OpenAI · Research', url: 'https://openai.com/index/why-language-models-hallucinate/' }
        ]
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: '18. RAG: Giving Models Access to Knowledge',
        description: 'RAG extends simple chat by letting models retrieve and use external documents—overcoming knowledge cutoffs and accessing private data.',
        paragraphs: [
            'Simple chat is limited to the model\'s training data (with a knowledge cutoff date) and has no access to your private documents. <strong>Retrieval-Augmented Generation (RAG)</strong> solves this by dynamically fetching relevant information and inserting it into the model\'s context before generating a response.',
            '<strong>How it works:</strong> (1) User asks a question. (2) System searches a document collection using <strong>semantic search</strong> (embeddings + vector databases—concepts from the architecture section). (3) Retrieved documents are injected into the prompt. (4) Model generates an answer informed by both its training and the retrieved text.',
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
        title: '19. Beyond Text: Multimodal Models & Tool Use',
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
            { type: 'article', title: 'Claude Vision Capabilities', meta: 'Anthropic · Multimodal', url: 'https://www.anthropic.com/news/claude-3-family' }
        ]
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: '20. Reasoning: Two Paradigms',
        description: 'Reasoning capability evolved from simple prompting tricks to fundamental architectural changes in how models think.',
        paragraphs: [
            'Early language models struggled with multi-step reasoning—they would jump to conclusions or make arithmetic errors. Two distinct approaches emerged to address this, representing fundamentally different philosophies about where "thinking" should happen.',
            '<strong>Chain-of-Thought (CoT) — 2022:</strong> Discovered by Google Research, CoT is a <strong>prompting technique</strong> where you ask the model to "think step by step" or show its work. By generating intermediate reasoning steps, models dramatically improve on math, logic, and complex questions. Accuracy on grade-school math jumped from ~20% to ~60% with CoT prompting on GPT-3. <strong>Key limitation:</strong> The reasoning is visible, token-inefficient, and requires user-side prompt engineering.',
            '<strong>Inference-Time Compute (Test-Time Scaling) — 2024-2025:</strong> A paradigm shift where reasoning is <strong>built into the model itself</strong>. Instead of immediately answering, the model generates <strong>hidden reasoning tokens</strong>—internal "thoughts" not shown to the user. It explores multiple solution paths, verifies steps, and backtracks from errors. Models like OpenAI\'s o1, o3, and DeepSeek-R1 can "think" for seconds or minutes before responding. Performance scales with thinking time—on competitive programming and math olympiad problems, these models approach human expert level.',
            'This shift represents a fundamental change: reasoning moved from <strong>prompt engineering</strong> (user responsibility) to <strong>model architecture</strong> (system capability). Modern reasoning models automatically invest compute in reasoning when problems are hard—they don\'t need to be asked to think step-by-step.',
            'Unlike training compute, which is fixed before deployment, <strong>test-time compute</strong> is spent at each query—more thinking time generally means better answers on hard problems.'
        ],
        bullets: [
            '<strong>CoT (2022):</strong> Visible reasoning, user controls format, token-inefficient, works with any model',
            '<strong>Inference-Time Compute (2024+):</strong> Hidden reasoning, model controls strategy, token-efficient output, requires specialized training',
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
        title: '21. Agentic Workflows',
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
        category: 'infer',
        badge: 'Conclusion',
        title: '22. Recap: Key Learnings',
        description: 'Understanding how AI models work transforms you from a passive user into an informed practitioner.',
        paragraphs: [
            'Modern AI isn\'t magic. It\'s a <strong>high-fidelity statistical mirror</strong> of human-created text, trained on trillions of tokens to predict plausible continuations. Understanding this—the frozen weights, tokenization quirks, training phases, and inference mechanics—transforms AI from a mysterious black box into a predictable, engineered system whose failures and capabilities make sense.',
            '<strong>Key Mental Models to Internalize:</strong> (1) <strong>Frozen Artifact</strong>—inference never teaches the model; corrections only affect the current context. (2) <strong>Plausibility Over Truth</strong>—the model outputs what statistically "sounds right," not what is factually correct. (3) <strong>Token-Level Reasoning</strong>—models don\'t see letters or words as humans do; tokenization explains why they struggle with spelling, counting characters, or reversing text. (4) <strong>Context is Everything</strong>—the prompt stack (system + history + user input) is the model\'s entire universe; once context fills up, early information vanishes.',
            '<strong>Practical Decision Framework:</strong> When should you use different techniques? <strong>Prompt engineering</strong> (simple, cheap, instant, but limited)—start here for most tasks. <strong>RAG</strong> (retrieval-augmented generation)—when you need factual accuracy, citations, or access to private/current data. <strong>Fine-tuning</strong> (expensive, permanent)—when you need specialized behavior or domain expertise that can\'t fit in context. <strong>Reasoning models</strong> (o1, DeepSeek-R1)—when accuracy matters more than speed/cost, especially for math, code, and logic-heavy tasks.'
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

    },
    {
        category: 'adv',
        badge: 'Looking Ahead',
        title: '23. Looking Ahead',
        description: 'A look beyond the core content: emerging architectures and research directions.',
        paragraphs: [
            'Transformers dominate today, but several directions are already in production or heavy research.',
            '<strong>JEPA (Joint Embedding Predictive Architecture):</strong> Yann LeCun\'s vision—predict in <em>representation space</em>, not pixels or tokens. Goal: sample-efficient learning, world models, planning. V-JEPA and I-JEPA are early implementations.',
            '<strong>Mamba & State Space Models (SSMs):</strong> Linear or near-linear sequence complexity instead of quadratic attention. Recurrent state, long context without the same memory cost. Used in some long-context and efficient LLMs.',
            '<strong>Mixture of Experts (MoE):</strong> Sparse activation—route each token to a subset of "expert" sub-networks instead of one dense stack. Lets you scale total parameters (e.g. 400B+) while keeping compute per token similar. Used in DeepSeek-V3, Llama-4, Gemini-2.5, Mixtral.',
            '<strong>RWKV & RetNet:</strong> RNN-like inference (O(1) memory) with parallelizable training. "Successor to Transformer" narrative; constant-memory decoding and long context. RWKV-7 and RetNet are in active use.',
            '<strong>Hybrids:</strong> Models that mix attention with SSMs and/or MoE (e.g. Jamba: attention + Mamba + MoE; Qwen3-Next, linear attention). "Attention was never enough"—combining mechanisms is a major trend.'
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
    }
];