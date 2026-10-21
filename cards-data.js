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
            'Broadly speaking, there are three main topics to cover: model architecture (how the machine is built), model inference (how it runs when you use it), and model training (how its weights were learned).',
            'At the heart of modern models is a concept called the <strong>Transformer</strong>, a type of neural network architecture originally designed for machine translation that now processes text, images, audio, and more. Transformers were defined in a seminal paper in 2017 by Vaswani et al. and have since become the de facto standard for language model architecture. Transformers unlocked the ability to train models with billions of parameters, which is what allows modern models to be so powerful.'
        ],
        bullets: [
            'Understanding this architecture helps predict failures, hallucinations, and informs effective use of tools like RAG',
            'The presentation flows: <strong>Architecture</strong> → <strong>Inference</strong> → <strong>Training</strong> → <strong>Advanced Capabilities</strong> (we see how the machine runs before we see how it was built)'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Training vs. Inference:</strong> Training updates the model\'s "brain" (weights)—a massive structure that can contain billions—sometimes hundreds of billions—of parameters. Inference is the act of querying that "frozen" brain. Chatting provides temporary context, but it does not permanently teach the model or update its knowledge base.'
        },
        resources: [
            { type: 'video', title: 'Generative AI in a Nutshell', meta: '18 min · Henrik Kniberg', url: 'https://www.youtube.com/watch?v=2IK3DFHRFfw' },
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
        title: '2. Feed-Forward Neural Networks (FFNN)',
        description: 'The "classic" neural network architecture where data flows in one direction—the ancestor of all modern AI.',
        paragraphs: [
            'FFNNs are the most basic form of neural network, rooted in the research around <strong>Perceptrons</strong> (1958). Feed-forward neural networks process information in one direction—input → hidden layers → output—with no loops or memory. The key breakthrough was <strong>backpropagation</strong> (popularized in 1986), which made it practical to train multilayer networks capable of learning complex, non-linear relationships. Truly <em>deep</em> networks took another two decades—better activation functions, more data, and GPUs arrived around 2010–2012.',
            'FFNNs are the foundation of deep learning, but they process each input independently. Because they have no memory of previous inputs, they struggle with sequential data like text or audio—a limitation that motivated the development of RNNs, and eventually the Transformer\'s Attention mechanism.',
            '<strong>FFNNs never went away.</strong> Every Transformer layer (Slide 8) contains a feed-forward network, and these blocks hold roughly two-thirds of a model\'s parameters. Research suggests they act like a vast learned lookup table—much of what a model "knows" about facts is stored here.',
            '<strong>Activation functions</strong> (like ReLU) are applied between layers to introduce non-linearity. Without them, stacking layers wouldn\'t add any benefit—the network would be no more powerful than a single layer. They enable networks to learn complex patterns.'
        ],
        bullets: [
            '<strong>One-Way Flow:</strong> Data flows forward only (input → output), no feedback loops',
            '<strong>No Memory:</strong> Each input processed independently, no context from previous inputs',
            '<strong>Backpropagation (1986):</strong> Enables training of multilayer networks',
            '<strong>Still Inside Every LLM:</strong> The feed-forward block in each Transformer layer is an FFNN',
            '<strong>Activation Functions:</strong> Introduce non-linearity so the network can learn complex patterns, not just linear relationships',
            '<strong>Limitation:</strong> Cannot model temporal dependencies or sequential patterns'
        ],
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
        title: '3. Tokens & Tokenization',
        description: 'Models don\'t read text directly—they process numeric token IDs that represent pieces of words.',
        paragraphs: [
            'AI models operate on numbers, not letters. A <strong>tokenizer</strong> converts text into integer IDs representing vocabulary fragments. For example, "Ingenious" might split into three tokens: <code>In</code>, <code>gen</code>, and <code>ious</code>.',
            'This approach (often <strong>BPE</strong>—<strong>Byte Pair Encoding</strong>—or similar) balances efficiency and flexibility: common words stay whole, rare words split into reusable parts.',
            'Modern implementations like GPT\'s tokenizer use byte-level BPE, which operates on raw bytes rather than characters, using the 256 possible byte values (0-255) as a universal foundation. This allows it to represent any language, emoji, or Unicode character through UTF-8 encoding, making it language-agnostic.',
            'Multimodal models do something similar for images (small patches) and audio (short chunks), turning each piece into a vector (Slide 18).'
        ],
        bullets: [
            'Token count determines cost and speed—more tokens = higher compute',
            '<strong>Largely Blind to Characters:</strong> The model doesn\'t see "c-a-t," it sees the single token <code>cat</code>. It can only learn spelling indirectly, which is why AI struggles with counting letters, spelling backwards, or other character-level tasks',
            '<strong>Tokens are pieces, not words:</strong> The model often sees subwords like <code>un</code> + <code>believ</code> + <code>able</code>',
            'BPE originated in data compression (1994) and was later adopted for NLP tokenization; modern tokenizers repurpose that idea.'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Tokens Are the Model\'s Atoms:</strong> To a tokenizer, <code>cat</code>, <code>&nbsp;cat</code> (with a leading space), and <code>Cat</code> can be three unrelated IDs. Everything the model ever "reads" or "writes" is a sequence of these IDs—text, tool calls, and even its hidden reasoning. Paste a sentence into the tokenizer tool below to see the splits for yourself.'
        },
        resources: [
            { type: 'tool', title: 'OpenAI Tokenizer', meta: 'Interactive · Try BPE', url: 'https://platform.openai.com/tokenizer' },
            { type: 'video', title: 'Byte Pair Encoding Explained', meta: 'Annie Sexton · Tokenization', url: 'https://www.youtube.com/watch?v=4A_nfXyBD08' },
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
        title: '4. From Numbers to Meaning: Embeddings & Latent Space',
        description: 'Token IDs are just numbers—embeddings transform them into rich, meaningful representations in a high-dimensional "Latent Space."',
        paragraphs: [
            'After tokenization, the model has a sequence of token IDs—integers like [4829, 2121, 8945]. But numbers alone are meaningless. The model needs to understand what each token <em>represents</em>.',
            'Enter <strong>embeddings</strong>: the model looks up each token ID in a learned table (the <strong>embedding matrix</strong>) and retrieves a <strong>vector</strong>—a list of hundreds or thousands of numbers. Together, those numbers encode the token\'s semantic properties and grammatical role. This idea—that meaning comes from context ("you shall know a word by the company it keeps," 1957)—led from early methods like Word2Vec (2013) to the powerful embeddings in modern Transformers.',
            'These vectors live in <strong>Latent Space</strong>, a high-dimensional coordinate system where similar meanings cluster together. A vector might have 4,096 dimensions. Meaning is stored in <em>directions</em> through this space rather than in individual dimensions—any single number on its own is usually uninterpretable, and models pack in far more concepts than they have dimensions (researchers call this <strong>superposition</strong>).',
            '<strong>Important:</strong> this lookup is <em>context-free</em>. The word "bank" gets the exact same starting vector in "river bank" and "bank account." Disambiguating it is the job of the Transformer layers that come next (Slides 8–9), which repeatedly blend in information from surrounding tokens.',
            'The model also needs to know word order. Without position information, "dog bites man" and "man bites dog" would contain the same set of vectors. The original Transformer <em>added</em> a positional encoding to each embedding before the first layer; most modern LLMs instead use <strong>rotary position embeddings (RoPE)</strong>, which inject position inside every attention layer.'
        ],
        bullets: [
            '<strong>Embedding Lookup:</strong> Token ID → retrieve learned vector from embedding table',
            '<strong>Latent Space:</strong> An organic "map of meaning" where distance = difference in concept',
            '<strong>Learned Features:</strong> Models discover their own directions of meaning during training—not hand-crafted by humans',
            '<strong>Static, Then Contextual:</strong> The lookup gives one vector per token regardless of context; the layers above make it context-aware',
            '<strong>Position:</strong> Order information is added to the input (original design) or injected inside each attention layer (RoPE, used by most modern LLMs)'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>The Flavor Profile Analogy:</strong> If tokenization assigns each word a locker number, embeddings are the contents—a profile describing the word\'s "flavor." Picture a direction in the space that means "sweetness" and another that means "spiciness." In a 4,096-dimensional space, the model builds a hyper-detailed profile for every concept. (The caveat: these flavors are blended across many numbers, not one flavor per number.)<br><br><strong>Latent Space Math:</strong> Take the vector for "King," subtract the vector for "Man," and add the vector for "Woman." The nearest word to the result is "Queen." The <em>difference</em> between man and woman behaves like a consistent direction you can add or subtract. This worked strikingly well in Word2Vec; in modern models the geometry is messier, but the core idea—concepts as directions—still holds.'
        },
        resources: [
            { type: 'video', title: 'Tokens vs. Embeddings', meta: 'Annie Sexton · Visual', url: 'https://www.youtube.com/watch?v=izbifbq3-eI' },
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
        title: '5. Recurrent Neural Networks (RNNs)',
        description: 'RNNs extend feed-forward networks with memory, enabling them to process sequential data.',
        paragraphs: [
            '<strong>Recurrent Neural Networks (RNNs)</strong> were developed in the mid-1980s (notably by Jordan in 1986 and Elman in 1990) to solve the fundamental limitation of feed-forward networks: their inability to handle sequential data. RNNs add a loop: the network\'s output at one step is fed back in as an input at the next step, creating a "memory" that carries forward through time.',
            'Unlike feed-forward networks, RNNs maintain a <strong>hidden state</strong> that carries information from previous time steps. At each step, the RNN takes the current input and the previous hidden state, combines them, and produces both an output and a new hidden state. This allows the network to remember and use information from earlier in the sequence.',
            'RNNs excel at tasks where order matters: language modeling, speech recognition, time series prediction. However, they have a critical flaw: the <strong>vanishing gradient problem</strong>. When processing long sequences, gradients shrink exponentially as they propagate backward through time. By the time an RNN reaches word 50, it has largely forgotten word 1, limiting their ability to capture long-range dependencies.'
        ],
        bullets: [
            '<strong>Sequential Processing:</strong> Must process tokens one at a time; step 2 cannot start until step 1 is finished',
            '<strong>Hidden State:</strong> Carries context from previous steps forward',
            '<strong>Memory:</strong> Can remember information from earlier in the sequence',
            '<strong>Vanishing Gradients:</strong> Information decays over long sequences'
        ],
        callout: {
            type: 'insight',
            content: '<strong>The Trade-off:</strong> RNNs gained memory at the cost of parallelization. They must process each sequence step by step, so a GPU cannot work on all positions of a sentence at once—a bottleneck that would eventually limit model scale.'
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
            '<strong>Long Short-Term Memory (LSTM)</strong> networks (Hochreiter &amp; Schmidhuber, 1997) were designed to solve RNNs\' vanishing gradient problem. They introduce a more sophisticated memory mechanism with three gates: <strong>forget gate</strong> (what to discard), <strong>input gate</strong> (what to store), and <strong>output gate</strong> (what to use).',
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
            { type: 'article', title: 'Understanding LSTM Networks', meta: 'Chris Olah · Classic explainer', url: 'https://colah.github.io/posts/2015-08-Understanding-LSTMs/' },
            { type: 'article', title: 'LSTM Networks | A Detailed Explanation', meta: 'Rian Dolphin · Towards Data Science', url: 'https://towardsdatascience.com/lstm-networks-a-detailed-explanation-8fae6aefc7f9/' }
        ],
        image: {
            url: 'https://towardsdatascience.com/wp-content/uploads/2020/10/1jikKbzFXCq-IYnFZankIMg.png',
            caption: 'LSTM cell architecture with forget, input, and output gates',
            attribution: 'Rian Dolphin · Towards Data Science'
        }
    },
    {
        category: 'arch',
        badge: 'Architecture',
        title: '7. Encoder-Decoder Architecture (Seq2Seq)',
        description: 'The "Seq2Seq" breakthrough introduced the "Thought Vector"—a way to turn a whole sentence into a single mathematical point.',
        paragraphs: [
            'RNNs and LSTMs process sequences, but they struggle with a key challenge: how do you transform one sequence into another sequence of different length? <strong>Encoder-decoder architecture</strong> (also called <strong>Seq2Seq</strong>) was introduced in 2014 in two independent papers—one by Sutskever, Vinyals, and Le at Google, and one by Cho and colleagues—to solve this.',
            'The architecture splits the task between two RNNs: The <strong>encoder</strong> processes the entire input sequence, compressing it into a single fixed-size vector in Latent Space called the <strong>context vector</strong> (often called a "Thought Vector"). The <strong>decoder</strong> then takes this single vector and "unpacks" it into a new sequence.',
            '<strong>Terminology note:</strong> "Thought Vector" was popular in early Seq2Seq literature but modern papers typically use "context vector" or "hidden state." If you search current research, look for "context vector" instead.',
            'The hope was that the "Thought Vector" would act as a <strong>universal coordinate</strong> for a concept. If you translate "The cat is on the mat" to French, the encoder places that "thought" at a spot in the Map of Meaning (Latent Space), and the decoder describes that spot using French words. In practice these vectors were only partly language-neutral—the dream of a perfect universal coordinate was an aspiration, not a result.',
            'This design revealed a critical limitation: the <strong>context bottleneck</strong>. Trying to compress a long, complex document into one single vector is like trying to summarize a whole book into one sentence—you lose the nuances.',
            '<strong>The fix (Bahdanau et al., 2014):</strong> let the decoder look back at <em>all</em> of the encoder\'s hidden states—one per input word—and choose which ones to focus on at each step. This was the birth of <strong>Attention</strong>, and it set up the question the next slide answers.'
        ],
        bullets: [
            '<strong>Context Vector:</strong> A single point in Latent Space representing an entire sequence\'s meaning',
            '<strong>Universal Coordinate (the aspiration):</strong> Ideally, the same "thought" sits in the same spot regardless of language—only partly true in practice',
            '<strong>Attention (2014):</strong> The decoder looks back at every input word instead of relying on one compressed vector',
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
            'A Transformer is a <strong>stack of identical layers</strong> (12, 24, 48, 96+). Think of it as a skyscraper: each floor performs the same two operations on every token. (1) <strong>Self-attention</strong> lets tokens communicate and update based on context. (2) A <strong>feed-forward network</strong> (the FFNN from Slide 2) processes each token independently, refining its meaning.',
            'Each token carries a running vector up the building, called the <strong>residual stream</strong>. Floors don\'t replace that vector; each one <em>reads</em> it, computes a small update, and <em>adds</em> the update back in. These residual connections (plus layer normalization) are what keep training stable across 100+ layers.',
            'Roughly speaking, early layers handle local patterns like word pieces and syntax, and middle layers build the most abstract representations of meaning. The final layers convert all of that into a prediction of the <em>next</em> token. These roles overlap a lot—they\'re tendencies researchers have measured, not clean departments.',
            'Most modern LLMs (GPT, Claude, Llama) are <strong>decoder-only Transformers</strong>: a single stack with one crucial rule, the <strong>causal mask</strong>. Each token can only look at itself and the tokens <em>before</em> it, never ahead. This is what lets the model learn by predicting the next token without "seeing the answer."',
            '<strong>Parallel or one-at-a-time? Both.</strong> When reading text (training, or reading your prompt), every position is processed simultaneously—the causal mask makes this safe. When <em>writing</em> a response, the model must still produce one token at a time, because each new token depends on the last (Slide 12).'
        ],
        bullets: [
            '<strong>Layer Pattern:</strong> (Normalize → Attention → Add) → (Normalize → Feed-forward → Add)',
            '<strong>Residual Stream:</strong> Each layer adds a small update to a running vector per token, rather than replacing it',
            '<strong>Causal Mask:</strong> Tokens see only the past — never future tokens',
            '<strong>Parallel Reading, Sequential Writing:</strong> All positions are processed at once for training and prompt reading; generation is still one token at a time',
            '<strong>Scale Matters:</strong> More layers, wider layers, and more training data together drive capability — depth alone is not the whole story'
        ],
        callout: {
            type: 'insight',
            content: '<strong>The Skyscraper Analogy:</strong> Each token rides an elevator (the residual stream) up dozens of floors. On every floor it stops, listens to the earlier tokens below and behind it (attention), consults the floor\'s reference library (feed-forward), and adds notes to the clipboard it carries. It never erases the clipboard; it only adds to it. On the ground floor, "bank" knows only its dictionary entry. By the middle floors, its notes say "river bank, not money." By the top floor, the notes have been turned into a guess about which word comes next.'
        },
        resources: [
            { type: 'video', title: 'Transformers, the tech behind LLMs', meta: 'Deep Learning Ch. 5 · 3Blue1Brown', url: 'https://www.youtube.com/watch?v=wjZofJX0v4M' },
            { type: 'video', title: 'Visualizing Transformers and Attention', meta: 'Talk · Grant Sanderson (3Blue1Brown)', url: 'https://www.youtube.com/watch?v=KJtZARuO3JY' },
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
            'Each token\'s Query is compared against the Keys of every token it is allowed to see—itself and all <em>earlier</em> tokens (the causal mask from Slide 8)—via a dot product. This produces <strong>attention scores</strong>: essentially a relevance ranking. <strong>Softmax</strong> then converts these raw scores into weights that sum to 1. High-relevance tokens get most of the weight; irrelevant tokens get nearly zero. The output is a weighted blend of those tokens\' Values, which is added into the token\'s residual stream.',
            '<strong>Example:</strong> In "We sat on the river bank," the word "bank" sends out its Query. When matched against the Keys of the earlier words, "river" scores high while "the" and "on" score low. So "bank" pulls mostly from "river\'s" Value, nudging its representation toward "riverbank" rather than "financial institution." Every token does this at the same time, each looking backward at its own context.',
            'Notice the word order matters: in "The bank by the river," the word "bank" <em>cannot</em> see "river" yet, because "river" comes later. The model resolves the meaning further along—at the token "river," which can look back at "bank."',
            'Models don\'t run this process once—they run it many times in parallel: 12 heads in small models, 64–128 in large ones. Each head has its own Q/K/V weight matrices (large models often share K and V across groups of heads to save memory). Different heads tend to pick up different relationships: one might track syntax, another the previous word, another which noun a pronoun refers to. The outputs are concatenated and mixed, giving the model multiple simultaneous perspectives on every token.'
        ],
        bullets: [
            '<strong>Learned Weights:</strong> Each layer\'s Q, K, V matrices are trained parameters — the model learns <em>what to pay attention to</em>',
            '<strong>Causal Mask:</strong> A token attends only to itself and earlier tokens — never to future ones',
            '<strong>Softmax:</strong> Converts raw scores to weights summing to 1 — the same function reappears at the very top of the model (Slide 13) to turn scores into next-token probabilities',
            '<strong>Multi-Head:</strong> Dozens of parallel attention operations per layer, each tending to specialize in different patterns',
            '<strong>Cost:</strong> Every token compares itself with every earlier token, so cost grows with the square of the context length — a key motivation for the alternatives on Slide 23'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>The Cocktail Party:</strong> Imagine you\'re at a crowded party (self-attention). You tune into the conversations most relevant to you (high attention scores) and tune out background noise (low scores). Now imagine 16 versions of you at the same party, each listening for something different — one for names, one for emotions, one for topics. That\'s multi-head attention: multiple simultaneous filters on the same scene.<br><br><em>With the architecture understood, let\'s see how these structures work when you use the model.</em>'
        },
        resources: [
            { type: 'video', title: 'Attention in Transformers', meta: '26 min · 3Blue1Brown', url: 'https://www.youtube.com/watch?v=eMlx5fFNoYc' },
            { type: 'article', title: 'The Illustrated Transformer', meta: 'Jay Alammar · Multi-head', url: 'https://jalammar.github.io/illustrated-transformer/' }
        ]
    },
    {
        category: 'infer',
        badge: 'Inference',
        title: '10. The Frozen State & Prompt Stack',
        description: 'After training, model weights are frozen. Inference uses a structured prompt stack to generate responses without learning.',
        paragraphs: [
            'Once training completes, the model\'s weights are <strong>locked</strong>. Inference (generating responses) reads these weights but never modifies them. This is why chatting doesn\'t teach the model anything permanent—corrections only affect the current conversation\'s context.',
            'When you send a message, the system assembles a <strong>prompt stack</strong>: (1) <strong>System Prompt</strong> (hidden instructions defining persona and rules), (2) <strong>Tool definitions and retrieved documents</strong>, if any (Slides 17 and 19), (3) <strong>Conversation History</strong> (prior messages re-sent on each turn), and (4) <strong>User Prompt</strong> (your message).',
            'The model doesn\'t receive these as separate boxes. They are flattened into <strong>one long sequence of tokens</strong>, with special marker tokens showing who said what—a <strong>chat template</strong>. Conceptually it looks like: <code>[system] You are a helpful assistant… [user] What\'s the capital of France? [assistant]</code>. The model\'s job is simply to continue this text from the point after <code>[assistant]</code>.',
            'This single sequence flows through the "skyscraper" of Transformer layers (Slide 8), and the model enters the <strong>Autoregressive Loop</strong> covered on Slide 12: generate one token, append it, generate the next.'
        ],
        bullets: [
            '<strong>Frozen Weights:</strong> Feedback doesn\'t update the model\'s "brain" permanently',
            '<strong>Weights vs. Activations:</strong> <strong>Weights</strong> are frozen long-term memory — the billions of parameters set during training. <strong>Activations</strong> are the transient numbers computed as your prompt flows through the network. Between turns, the conversation survives only as <em>text</em> that is sent again',
            '<strong>Context Window:</strong> The fixed maximum number of tokens (input + output) the model can process in one request',
            '<strong>Lost in the Middle:</strong> Research (2023) found models used information at the start and end of long prompts better than information in the middle. Newer models have improved, but placing key material first or last is still good practice',
            '<strong>"Memory" Features:</strong> When a chat app "remembers" you across conversations, it is saving notes and inserting them into the prompt stack — the weights are still unchanged'
        ],
        callout: {
            type: 'note',
            content: '<strong>The Context Bottleneck:</strong> Because the entire history is sent with every turn, long conversations get slower and more expensive. The model itself never quietly "forgets." When a conversation outgrows the context window, the <em>application</em> decides what to do: reject the request, drop the oldest messages, or replace them with a summary. Whatever is cut is simply gone from the model\'s view.'
        },
        resources: [
            { type: 'article', title: 'Claude Prompting Best Practices', meta: 'Anthropic · Prompt engineering', url: 'https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices' },
            { type: 'article', title: 'Lost in the Middle: How Language Models Use Long Contexts', meta: 'Liu et al. · 2023 paper', url: 'https://arxiv.org/abs/2307.03172' },
            { type: 'video', title: 'Context Engineering Explained', meta: '17 min', url: 'https://www.youtube.com/watch?v=p6s82Ax8yrs' },
            { type: 'article', title: 'Andrew Mayne Prompts', meta: 'Prompt engineering techniques & history', url: 'https://andrewmayneprompts.com/' }
        ]
    },
    {
        category: 'infer',
        badge: 'Inference',
        title: '11. Preparing the Input',
        description: 'Before the model can generate anything, your message must be assembled, tokenized, and embedded.',
        paragraphs: [
            'When you press "Send," a precise sequence of operations begins. The first three steps prepare the raw input before any generation can happen.',
            '<strong>Step 1 — Prompt Assembly:</strong> The system combines hidden instructions (system prompt), prior conversation turns (history), and your new message into a single text block. This assembled prompt is the model\'s <em>entire</em> view of the conversation.',
            '<strong>Step 2 — Tokenization:</strong> The full prompt is split into token IDs (Slide 3). A short question might become 20 tokens; a long conversation with system instructions might be 4,000+. Each token costs compute and counts against the context window.',
            '<strong>Step 3 — Embedding & Position:</strong> Each token ID is converted into a vector (Slide 4), and position information is attached so the model knows word order. The result is a sequence of vectors ready to enter the Transformer.'
        ],
        bullets: [
            '<strong>Every Turn Re-Sends Everything:</strong> Each new message sends the full prompt (system + history + your message) again — the model keeps nothing from the last turn',
            '<strong>Read Once per Turn:</strong> Within a turn, the prompt is tokenized and read just once, with all positions processed in parallel ("prefill"). Generated tokens are already token IDs, so they are never re-tokenized',
            '<strong>Cost Scales with Length:</strong> Longer conversations = more input tokens to read, and each generated token must attend over a longer history',
            '<strong>The Context Window Limit:</strong> The prompt plus the response must fit within the context window; beyond that, the application must trim or summarize'
        ],
        callout: {
            type: 'note',
            content: '<strong>Prep Once, Then Loop:</strong> These three steps happen once at the start of each turn. After that, the model enters a generation loop (next slide) where each new token is fed straight back in as an ID—no re-assembly, no re-tokenizing. Providers can even cache the processed start of a prompt (such as a long system prompt) across turns, which is why repeated prefixes are often billed at a discount.'
        },
        resources: [
            { type: 'article', title: 'Prompt Caching', meta: 'Anthropic · Docs', url: 'https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching' }
        ]
    },
    {
        category: 'infer',
        badge: 'Inference',
        title: '12. The Forward Pass & Autoregressive Loop',
        description: 'The prepared vectors flow through the Transformer, a token is selected, and the process loops.',
        paragraphs: [
            '<strong>Step 4 — The Transformer Forward Pass:</strong> The full sequence of vectors flows upward through every Transformer layer (the "skyscraper" from Slide 8). Attention lets tokens communicate; feed-forward networks refine meaning. After dozens of layers, the <em>final</em> token\'s vector emerges at the top, encoding the model\'s best prediction for what comes next.',
            '<strong>Step 5 — Token Selection:</strong> That final vector is compared against the entire vocabulary to produce scores (logits). A token is selected (details in the next slide) and streamed to you.',
            '<strong>Step 6 — The Loop:</strong> The selected token is appended to the sequence, and the model runs again to predict the following token. This is the <strong>autoregressive loop</strong>—one token at a time until the model produces a stop token or hits the output limit.',
            '<strong>The mental model vs. what really runs.</strong> The simple mental model is "re-read the whole sequence for every new token," and the answer would be mathematically the same. In practice that would be far too slow. Because of the causal mask (Slide 8), earlier tokens never change when later ones are added. So the system saves each earlier token\'s Key and Value vectors (the <strong>KV cache</strong>), and each new token makes just <em>one</em> pass up the skyscraper, looking back at the cached history.'
        ],
        bullets: [
            '<strong>Why Streaming Exists:</strong> Each token is available as soon as it\'s generated, so you see text appear word-by-word rather than waiting for the full response',
            '<strong>Two Phases:</strong> <em>Prefill</em> reads your whole prompt in parallel (fast); <em>decode</em> generates the answer one token at a time (slower). This is why long prompts are cheap compared with long answers',
            '<strong>KV Cache:</strong> Stored Keys and Values of earlier tokens let each new token be computed in a single pass — the cache lives only for the current request',
            '<strong>One Token at a Time:</strong> Generation is sequential even in a Transformer — each token depends on the one before it'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Key Insight:</strong> At its core, the whole system has one skill: <em>predict the next token</em>. Writing an essay, calling a tool (Slide 19), and "thinking" before answering (Slide 20) are all this same loop running many times. That single skill can look like reasoning, writing, or coding—but it is still prediction, not guaranteed truth or perfect calculation.'
        },
        resources: [
            { type: 'video', title: 'Let\'s build GPT: from scratch, in code, spelled out', meta: '2 hrs · Andrej Karpathy', url: 'https://www.youtube.com/watch?v=kCc8FmEb1nY' }
        ]
    },
    {
        category: 'infer',
        badge: 'Inference',
        title: '13. The Selection Dice Roll',
        description: 'The final step: turning a refined vector back into a human word.',
        paragraphs: [
            'At the roof of the skyscraper, the model holds a highly refined vector representing the "concept" of what should come next. To turn this back into a word, the model multiplies this vector against the <strong>unembedding matrix</strong>—calculating a dot-product score (<strong>logit</strong>) for every token in its vocabulary. The unembedding matrix holds one "output direction" per token; a high logit means the model\'s final vector points strongly in that token\'s direction.',
            'Some models (like GPT-2) use <strong>weight tying</strong>: the input embedding table doubles as the output table, so a token\'s output direction is the same as its input embedding. Many larger models keep the two tables separate, giving the model more freedom in how it represents inputs versus predictions. Either way, this step translates the model\'s internal state back into the language of tokens.',
            '<strong>Example:</strong> Given "The Renaissance banking family that sponsored Michelangelo was the ___," the logits are converted into a probability distribution via <strong>softmax</strong>. The model might assign "Medici" 75%, "Borgia" 5%, and spread the rest over thousands of other tokens. It doesn\'t "know" the answer is Medici; it has learned that Medici is by far the most likely continuation in this context.',
            '<strong>Temperature</strong> reshapes that distribution before a token is drawn (mathematically, the logits are divided by the temperature before softmax). Temperature 1.0 uses the model\'s distribution as learned. Lower values sharpen it toward the top choice; at 0 the model always takes the single most likely token (called "greedy" decoding—predictable, but prone to repetitive loops and no more truthful). Values above 1.0 flatten it, giving unlikely tokens more of a chance.',
            'Most systems also trim the long tail before sampling: <strong>top-k</strong> keeps only the k most likely tokens, and <strong>top-p</strong> (nucleus sampling) keeps the smallest set of tokens whose probabilities add up to, say, 90%. This prevents a rare nonsense token from ever being drawn.'
        ],
        bullets: [
            '<strong>Logits:</strong> Raw dot-product scores between the final vector and every token\'s row in the unembedding matrix',
            '<strong>Weight Tying:</strong> Some models reuse the embedding table as the unembedding table — the same table works in both directions',
            '<strong>Temperature:</strong> Sharpens (below 1) or flattens (above 1) the probability distribution before sampling',
            '<strong>Top-k / Top-p:</strong> Cut off the unlikely tail so only sensible candidates can be drawn',
            '<strong>Why Answers Vary:</strong> With temperature above 0, the same prompt can produce different tokens each time — this is the main reason the same question gets different answers',
            '<strong>Probabilistic:</strong> The model is optimized for "plausibility," not necessarily truth'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Prediction, not Truth:</strong> If a confident-sounding but wrong continuation is the most likely one, the model will produce it—not because it "wants" to lie, but because that is what its learned probabilities favor. Research also suggests training and benchmarks have historically rewarded a confident guess over "I don\'t know," which pushes models toward guessing. It is a statistical mirror, not a database.<br><br><em>We\'ve seen how the machine runs — the forward pass through a frozen model. Next: how we built the machine. The difference between using a map and drawing the map.</em>'
        },
        resources: [
            { type: 'video', title: 'Why LLMs Hallucinate', meta: 'Practical · Video', url: 'https://www.youtube.com/watch?v=cfqtFvWOfg0' },
            { type: 'article', title: 'Why language models hallucinate', meta: 'OpenAI · Research', url: 'https://openai.com/index/why-language-models-hallucinate/' }
        ]
    },
    {
        category: 'train',
        badge: 'Training',
        title: '14. Pre-Training',
        description: 'Pre-training is where models learn the patterns, facts, and structures of human knowledge from massive text datasets.',
        paragraphs: [
            'During pre-training, the model consumes <strong>trillions of tokens</strong> from books, websites, research papers, and code repositories. The training objective is simple: predict the next token. Every prediction is scored by how much probability the model gave to the token that actually came next (the <strong>loss</strong>). Even a mostly-right guess is nudged to be a little more confident, and a badly wrong one is nudged a lot. These tiny weight adjustments are computed via <strong>backpropagation</strong>. Every learnable parameter from the architecture section — the embedding matrix, the Q/K/V attention matrices, the feed-forward weights in every layer — is tuned through this process. Nothing is hand-programmed; the model discovers what to pay attention to and how to process meaning entirely from data.',
            '<strong>Why training is parallel:</strong> Thanks to the causal mask (Slide 8), one pass over a 4,000-token document yields 4,000 predictions at once—each position predicts its next token while seeing only the tokens before it. This is exactly the parallelism RNNs lacked, and it is why Transformers could be trained on trillions of tokens.',
            'The same backpropagation algorithm popularized in 1986 (see Slide 2) now runs across thousands of GPUs simultaneously, allowing the model to learn from errors at an astronomical scale.',
            'This process takes months on thousands of GPUs, and frontier training runs now cost tens to hundreds of millions of dollars. Knowledge ends up spread across billions of weights as compressed patterns—not stored as a searchable list of facts. The result? A <strong>base model</strong> that can complete sentences, generate code, and recall facts—but often produces rambling or unhelpful outputs.'
        ],
        bullets: [
            '<strong>Scaling Laws:</strong> Prediction error falls smoothly and predictably as parameters, data, and compute grow together—with diminishing returns. A 2022 study ("Chinchilla") found roughly 20 training tokens per parameter is the compute-efficient balance',
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
            { type: 'video', title: 'What is Backpropagation?', meta: '14 min · 3Blue1Brown', url: 'https://www.3blue1brown.com/lessons/backpropagation' },
            { type: 'video', title: 'Deep Dive into LLMs like ChatGPT', meta: '3.5 hrs · Andrej Karpathy · Pre- and post-training', url: 'https://www.youtube.com/watch?v=7xTGNNLPyMI' }
        ],
        image: {
            url: 'https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4c51db77-8d97-45a9-bd2c-d71e930ff0b8_2292x1234.png',
            caption: 'LLM training stages: Pre-training, SFT, and RLHF',
            attribution: 'Cameron R. Wolfe · Deep Learning Focus'
        }
    },
    {
        category: 'train',
        badge: 'Training',
        title: '15. Post-Training',
        description: 'Post-training transforms a knowledgeable but unruly base model into a helpful, safe, and aligned assistant.',
        paragraphs: [
            'Base models know a lot but don\'t behave like assistants. Ask one "What is the capital of France?" and it may continue with "What is the capital of Germany?"—as if completing a quiz—because it is still just continuing text. It may also produce offensive content or ramble endlessly. <strong>Post-training</strong> teaches it to be a useful assistant through several techniques:',
            '<strong>Supervised Fine-Tuning (SFT):</strong> The model is trained on many examples of ideal conversations—written by humans, and increasingly generated and filtered by other models. It learns the assistant format and helpful behavior.',
            '<strong>Reinforcement Learning from Human Feedback (RLHF):</strong> Humans rank multiple model responses (A > B > C). A separate <strong>reward model</strong> is trained on these preferences to score new outputs. Then reinforcement learning optimizes the language model to produce responses that score highly, with a penalty that keeps it from drifting too far from its original behavior. This works but is complex, and models can learn to exploit flaws in the reward model ("reward hacking").',
            '<strong>Direct Preference Optimization (DPO):</strong> A simpler alternative that skips the reward model—it directly adjusts the model to prefer the better response in each ranked pair. Widely used, especially for open models.',
            '<strong>AI Feedback (RLAIF / Constitutional AI):</strong> Instead of humans ranking every pair, a model judges responses against a written set of principles (a "constitution"). This scales feedback far beyond what human labelers can provide.',
            '<strong>Reinforcement Learning with Verifiable Rewards (RLVR):</strong> For math, coding, and logic, the answer can be checked automatically (does the code pass the tests?). The model attempts thousands of problems, and attempts that reach correct answers are reinforced. This is how <strong>reasoning models</strong> (Slide 20) learn to think at length before answering.'
        ],
        bullets: [
            '<strong>SFT:</strong> Learn the assistant format by imitating example conversations',
            '<strong>Preference Learning (RLHF, DPO, RLAIF):</strong> Learn which of two responses is better—from human or AI judges',
            '<strong>Verifiable Rewards (RLVR):</strong> Learn to reason by being rewarded for correct, checkable answers',
            'Post-training instills safety guardrails (refusing harmful requests) and conversational norms like being concise and admitting uncertainty',
            '<strong>Side Effects:</strong> Preference training can make models <em>sycophantic</em> (telling people what they want to hear) and less diverse in their outputs'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Unlocking, Not Just Restricting:</strong> Post-training is often described as a "safety tax," but mostly it <em>unlocks</em> capability the base model already had. In OpenAI\'s 2022 InstructGPT study, people preferred the outputs of a 1.3B-parameter post-trained model over the 175B-parameter base GPT-3. The real trade-offs are subtler: occasional over-refusal of harmless requests, sycophancy, and a narrower range of styles.'
        },
        resources: [
            { type: 'article', title: 'Understanding and Using Supervised Fine-Tuning (SFT)', meta: 'Cameron R. Wolfe · Deep Learning Focus', url: 'https://cameronrwolfe.substack.com/p/understanding-and-using-supervised' },
            { type: 'article', title: 'Training Language Models to Follow Instructions (InstructGPT)', meta: 'Ouyang et al. · 2022 paper', url: 'https://arxiv.org/abs/2203.02155' },
            { type: 'video', title: 'RLHF, Clearly Explained', meta: '18 min · StatQuest', url: 'https://www.youtube.com/watch?v=qPN_XZcJf_s' },
            { type: 'video', title: 'RLHF in 4 Minutes', meta: '4 min · Sebastian Raschka', url: 'https://www.youtube.com/watch?v=vJ4SsfmeQlk' }
        ],
        image: {
            url: 'https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F680ffa81-7b96-474f-832b-4be758e8d2e6_1176x638.png',
            caption: 'Post-training pipeline: SFT, RLHF, and DPO alignment techniques',
            attribution: 'Cameron R. Wolfe · Deep Learning Focus'
        }
    },
    {
        category: 'train',
        badge: 'Training',
        title: '16. Bias, Fairness & Limitations',
        description: 'AI models inherit the gaps and perspectives of their training data—they are mirrors, not arbiters of truth.',
        paragraphs: [
            'Training data is a massive scrape of human history, which includes our biases, stereotypes, and cultural blind spots. Models learn these patterns with the same mathematical rigor they use to learn grammar. If the data overrepresents a specific worldview, the model will treat that worldview as the "default" statistical reality.',
            'Post-training alignment (RLHF/DPO) can act as a safety filter, teaching the model to refuse to generate explicit hate speech or harmful content. However, this does not "fix" the underlying bias. A model may still reflect subtle stereotypes in resume screening, healthcare suggestions, or creative writing because those patterns are baked into its billions of parameters.',
            'This creates a "Fluency Trap." Because the model is so articulate, we tend to attribute objective truth to its outputs. In reality, it is a statistical mirror. It doesn\'t have a moral compass or a fact-checking department; it has a map of how humans have spoken in the past.'
        ],
        bullets: [
            '<strong>Representation Bias:</strong> Dominant cultures and languages are better represented than minority ones.',
            '<strong>The Fluency Trap:</strong> High linguistic quality can mask factual errors or biased reasoning.',
            '<strong>Mitigation vs. Cure:</strong> Alignment reduces harmful outputs but rarely eliminates latent bias.',
            '<strong>User Responsibility:</strong> Human oversight is non-negotiable in high-stakes domains like hiring, legal, or medical.'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Bias Can Come From Training, Too:</strong> Not all bias comes from the data. Preference training (Slide 15) rewards answers people rate highly, and people tend to rate agreement highly. The result is <strong>sycophancy</strong>: a tendency to agree with the user, flatter, or back down when challenged. Ask for critique explicitly, and be wary when a model simply confirms what you already believe.<br><br><strong>No Silver Bullet:</strong> Bias mitigation is an ongoing process, not a solved problem. Even the most carefully trained models can produce biased outputs. The goal is harm reduction and transparency, not perfection. Always apply human judgment, especially in consequential decisions.<br><br><em>Training is complete. Next: advanced capabilities that extend this foundation.</em>'
        },
        resources: [
            { type: 'video', title: '3 Types of Bias in AI', meta: 'Google · Machine learning', url: 'https://www.youtube.com/watch?v=59bMh59JQDo' },
            { type: 'article', title: 'NIST AI 600-1: Generative AI Profile', meta: 'NIST · Risk management for generative AI (PDF)', url: 'https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf' }
        ]
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: '17. RAG & Data Security',
        description: 'RAG transforms models from "closed-book" memory engines into "open-book" practitioners with access to your private data.',
        paragraphs: [
            'Standard models are frozen in time; they only know what they learned during training. <strong>Retrieval-Augmented Generation (RAG)</strong> allows the model to "look things up" in your private document collection. When you ask a question, the system finds the relevant paragraphs in your database and feeds them to the model as "required reading" for its answer.',
            '<strong>How the lookup works</strong> (this is Slide 4 again): ahead of time, your documents are split into chunks, and each chunk is turned into an embedding vector by a dedicated <strong>embedding model</strong>, then stored in a vector database. When a question arrives, it is embedded the same way, and the system retrieves the chunks whose vectors sit closest to it in latent space—chunks about the same <em>meaning</em>, even if they use different words. Many systems combine this with ordinary keyword search.',
            'RAG effectively turns the model into a researcher with a library card. It doesn\'t have to guess from its own potentially outdated memory; it can quote specific sources and ground answers in your organization\'s documents—so long as retrieval finds the right ones.',
            '<strong>What RAG does and doesn\'t do for security:</strong> Because documents are supplied at inference time, they don\'t change the model\'s weights; the model doesn\'t permanently learn them. But with a hosted model, the retrieved text <em>is sent to the provider</em> as part of every prompt. Whether it is stored or used for training is governed by the provider\'s data policy and your contract—not by RAG itself. Keeping data fully in-house requires a self-hosted model.'
        ],
        bullets: [
            '<strong>Knowledge Cutoff:</strong> RAG gets around the training cutoff—answers are as current as the document index.',
            '<strong>Semantic Search:</strong> Retrieval finds text by meaning using embeddings, often combined with keyword search.',
            '<strong>Reduced Hallucination:</strong> Models are less likely to make things up when the answer is in front of them—but bad retrieval leads to confident wrong answers.',
            '<strong>Citations:</strong> RAG lets the model point to its source documents; citations should still be spot-checked.',
            '<strong>New Risks:</strong> Retrieval must respect user permissions (or it can surface documents a user shouldn\'t see), and text inside retrieved documents can carry hidden instructions to the model (<strong>prompt injection</strong>).'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>Analogy:</strong> RAG is like allowing a student to bring textbooks into an exam. They still use reasoning and comprehension—but can look up specific facts instead of guessing from memory. The better the textbooks (documents) and search skills (retrieval), the better the answers.'
        },
        resources: [
            { type: 'video', title: 'What is RAG?', meta: '6 min · IBM', url: 'https://youtube.com/watch?v=T-D1OfcDW1M' },
            { type: 'article', title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks', meta: 'Lewis et al. · 2020 original RAG paper', url: 'https://arxiv.org/abs/2005.11401' },
            { type: 'article', title: 'OpenAI Embeddings Guide', meta: 'Technical docs · Foundation for RAG', url: 'https://platform.openai.com/docs/guides/embeddings' },
            { type: 'video', title: 'Embeddings Explained', meta: '18 min · 3D visualizations', url: 'https://www.youtube.com/watch?v=eUbKYEC0D3Y' },
            { type: 'article', title: 'Building an End-to-End RAG Pipeline', meta: 'Chitika · From Ingestion to Generation', url: 'https://www.chitika.com/building-end-to-end-rag/' }
        ],
        image: {
            url: 'https://www.chitika.com/content/images/2025/02/image2-7.jpg',
            caption: 'End-to-end RAG pipeline: from data ingestion to generation',
            attribution: 'Chitika'
        }
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: '18. Multimodal Models: Richer Perception',
        description: 'Models now see images, hear audio, and watch video — by projecting every modality into the same embedding space as text.',
        paragraphs: [
            'Simple text chat is just the beginning. Modern models can understand images, audio, and video alongside text. The key idea is a <strong>shared input space</strong>: an image is cut into small square patches, and a <strong>vision encoder</strong> turns each patch into a vector. Audio is typically converted into a spectrogram and encoded the same way. A small adapter then projects these vectors into the same space the language model uses for its text embeddings, so they can sit in the same sequence as your words.',
            'This works because the Transformer doesn\'t care what its inputs represent—it only operates on vectors. If you can convert any input into a sequence of vectors, the same attention and feed-forward machinery can process it alongside text. This is why multimodal capabilities expanded so rapidly: the hard part (the Transformer) was already solved.',
            'A related technique, <strong>contrastive training</strong> (e.g., OpenAI\'s CLIP, 2021), explicitly trains images and their captions to land at nearby points, so a photo of a cat and the phrase "a photo of a cat" end up close together. This powers image search and many vision encoders.',
            '<strong>Understanding vs. generating:</strong> Reading an image and creating one are different jobs. Most image generators (DALL-E 3, Midjourney, Stable Diffusion) are <strong>diffusion models</strong>, which start from random noise and refine it step by step toward an image matching the text. Some newer systems generate images directly within a multimodal LLM.',
            '<strong>Examples:</strong> <strong>GPT-4o</strong>, <strong>Claude</strong>, and <strong>Gemini</strong> accept images (GPT-4o also handles audio; Gemini handles video and audio). <strong>Whisper</strong> is a dedicated speech-to-text Transformer. <strong>DALL-E</strong> and <strong>Midjourney</strong> generate images.'
        ],
        bullets: [
            '<strong>Encoders + Adapter:</strong> A vision or audio encoder turns raw input into vectors, then an adapter projects them into the language model\'s input space',
            '<strong>Same Core Machinery:</strong> Once inputs are vectors, the same Transformer layers process images and text together',
            '<strong>Contrastive Alignment (CLIP):</strong> Trains matching images and captions to land near each other in a shared space',
            '<strong>Generation Is Different:</strong> Most image generators are diffusion models, not next-token predictors',
            '<strong>Use Cases:</strong> Analyze charts/diagrams, describe images for accessibility, transcribe meetings, generate creative visuals, ask questions about photos'
        ],
        callout: {
            type: 'analogy',
            content: '<strong>The Flavor Profile Returns:</strong> Remember embeddings from Slide 4? Multimodal models extend the same idea. The vision encoder works like a translator that writes a "flavor profile" for each patch of an image in the same format the language model uses for words. Once translated, the language model can reason about the picture with the same machinery it uses for text—though it still knows which parts came from an image.'
        },
        resources: [
            { type: 'video', title: 'Coding a Multimodal (Vision) Language Model from Scratch', meta: 'Umar Jamil · Advanced · PyTorch walkthrough', url: 'https://www.youtube.com/watch?v=vAmKB7iPkWw' },
            { type: 'article', title: 'Claude Vision Capabilities', meta: 'Anthropic · Multimodal', url: 'https://docs.anthropic.com/en/docs/build-with-claude/vision' }
        ]
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: '19. Tool Use: From Words to Actions',
        description: 'Tool use lets models step beyond their frozen weights to interact with calculators, APIs, databases, and the real world.',
        paragraphs: [
            'LLMs are prediction engines, not calculators or databases. Their weights are frozen — they can\'t look up today\'s stock price, and long exact calculations are unreliable. <strong>Tool use</strong> (also called <strong>function calling</strong>) solves this by letting the model request external actions during inference.',
            'The mechanism is surprisingly simple, and it is still just next-token prediction. The prompt stack (Slide 10) includes descriptions of the available tools. When a tool would help, the model generates tokens that spell out a structured request, like <code>{"tool": "calculator", "input": "sqrt(144)"}</code>, followed by a special stop marker. The model itself executes nothing. <em>Your application</em> spots the request, runs the tool, and appends the result to the conversation. The model then continues, now able to "see" the result.',
            'This creates a loop where the model can decide which tool to use, call it, observe the result, and continue. Post-training (Slide 15) is what teaches models when a tool call is appropriate and how to format one correctly. An open standard, the <strong>Model Context Protocol (MCP)</strong>, now lets one tool integration work across many different AI applications.',
            '<strong>Examples:</strong> Web search for current information, code interpreters for data analysis, API calls for booking flights, database queries for customer records. Modern models can chain multiple tools sequentially to accomplish complex tasks — and this chaining is exactly what powers the agentic workflows we\'ll see next.'
        ],
        bullets: [
            '<strong>Why Tools Matter:</strong> Models can\'t reliably do long exact math, access real-time data, or interact with external systems on their own — tools extend their capabilities beyond text generation',
            '<strong>Still Just Tokens:</strong> A tool call is text the model generates; the surrounding application is what actually takes the action',
            '<strong>Use Cases:</strong> Precise calculations, real-time data (weather, stock prices), code execution, database access, workflow automation',
            '<strong>Structured Output:</strong> The model generates JSON or function-call syntax that your application intercepts and executes',
            '<strong>Tool Chaining:</strong> Models can use multiple tools in sequence — search the web, then summarize, then email the result'
        ],
        callout: {
            type: 'insight',
            content: '<strong>Stepping Out of the Frozen State:</strong> Tool use is the model\'s way of reaching beyond its frozen weights. It can\'t update its own knowledge, but it can ask an external system to look something up, run a calculation, or take an action. This transforms models from "text conversationalists" into systems that can act on the world.'
        },
        resources: [
            { type: 'article', title: 'Tool Use with Claude', meta: 'Anthropic · Function calling', url: 'https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview' },
            { type: 'article', title: 'Introduction to the Model Context Protocol', meta: 'MCP · Open standard for tools', url: 'https://modelcontextprotocol.io/docs/getting-started/intro' }
        ]
    },
    {
        category: 'adv',
        badge: 'Advanced',
        title: '20. Reasoning: Two Paradigms',
        description: 'Reasoning capability evolved from a prompting trick to a skill trained directly into models—using the same Transformer architecture.',
        paragraphs: [
            'Early language models struggled with multi-step reasoning—jumping to conclusions or making arithmetic errors. Two approaches emerged, representing different philosophies about where "thinking" should happen.',
            '<strong>Chain-of-Thought (CoT) — 2022:</strong> A <strong>prompting technique</strong> where you ask the model to "think step by step." By generating intermediate reasoning steps, models dramatically improve on math, logic, and complex questions—but the reasoning is visible, uses more output tokens (since reasoning steps appear in the response), and requires user-side prompt engineering.',
            '<strong>Why writing it out helps:</strong> Each token gets one trip up the skyscraper (Slide 12), which is a fixed amount of computation. A hard problem may need more steps than one trip allows. Writing out intermediate steps lets the model use its own earlier tokens as a scratchpad, spreading the work across many trips.',
            '<strong>Reasoning Models (2024+):</strong> Reasoning <strong>trained into the model itself</strong>. The architecture is the same Transformer; what changed is post-training. Using reinforcement learning on problems with checkable answers (Slide 15), models learn to produce long reasoning passages before answering—exploring approaches, checking their work, and backtracking from errors. Models like o1, DeepSeek-R1, and Claude with extended thinking can "think" for seconds or minutes. This shifts responsibility from <strong>prompt engineering</strong> (user) to <strong>training</strong> (model developer).',
            'These "thinking" tokens are ordinary tokens from the same autoregressive loop. Whether you can see them is a product decision: some providers hide or summarize them, while others, like DeepSeek-R1, show them in full.'
        ],
        bullets: [
            '<strong>CoT (2022):</strong> Visible reasoning, user controls format, uses more output tokens, works with any model',
            '<strong>Reasoning Models (2024+):</strong> Model decides how long to think, reasoning may be hidden or summarized, high total compute, requires specialized training (RL with verifiable rewards)',
            '<strong>Same Architecture:</strong> Reasoning models are standard Transformers — the difference is in how they are trained',
            '<strong>Faithfulness Caveat:</strong> A written-out chain of thought does not always reflect how the model actually reached its answer — treat it as helpful evidence, not a guaranteed explanation',
            '<strong>Performance Trade-off:</strong> Reasoning models are slower and more expensive per query, but dramatically more accurate on complex tasks',
            '<strong>Scaling Law:</strong> Test-time compute scales inference—more thinking time → better results (unlike training compute, which scales the model)'
        ],
        callout: {
            type: 'insight',
            content: '<strong>The Evolution:</strong> CoT proved that models <em>can</em> reason if given space to think. Reasoning models took the next step: teaching models <em>when and how</em> to allocate that thinking automatically. This mirrors human cognition: <strong>System 1</strong> (fast, intuitive, "autopilot") vs. <strong>System 2</strong> (slow, deliberate, "effortful thinking"). Reasoning models represent AI\'s first true "pause and think" capability.'
        },
        resources: [
            { type: 'video', title: 'Chain-of-Thought Prompting Explained', meta: 'CodeEmporium · 2022 breakthrough', url: 'https://www.youtube.com/watch?v=AFE6x81AP4k' },
            { type: 'article', title: 'DeepSeek-R1: Incentivizing Reasoning via Reinforcement Learning', meta: 'DeepSeek · 2025 paper · How reasoning is trained', url: 'https://arxiv.org/abs/2501.12948' },
            { type: 'article', title: 'Reasoning Models Don\'t Always Say What They Think', meta: 'Anthropic · Faithfulness research', url: 'https://www.anthropic.com/research/reasoning-models-dont-say-think' },
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
            '<strong>The ReAct Pattern (Reasoning + Acting, 2022):</strong> The core framework is simple but powerful: <strong>Thought → Action → Observation → Repeat</strong>. Example: debugging failing code. <em>Thought:</em> "The test fails with <code>NameError: name \'requests\' is not defined</code>—need to check the imports." <em>Action:</em> Read the file. <em>Observation:</em> "There\'s no <code>import requests</code> line." <em>Thought:</em> "Add the import." <em>Action:</em> Edit file. <em>Observation:</em> "Test now passes." The model iteratively adjusts its plan based on real feedback.',
            '<strong>Under the hood, nothing new:</strong> An agent is the tool-use loop from Slide 19 running many times. The model is still frozen. Its "memory" is whatever is in the context window plus any notes or files the agent writes and reads back later. When a task outgrows the context window, the agent must summarize or offload what it has learned.',
            '<strong>Production Systems:</strong> Coding agents (Claude Code, OpenAI Codex, Cursor, Devin), research tools, and customer-support bots are mostly built by their companies on their own agent loops. General-purpose frameworks such as <strong>LangGraph</strong>, <strong>CrewAI</strong>, and the official agent SDKs from model providers help teams build their own. Agents excel at <strong>planning</strong> (task decomposition), <strong>memory</strong> (maintaining state), and <strong>reflection</strong> (self-verification). However, they\'re expensive (dozens of LLM calls per task), can be unreliable (stuck in loops, wrong tool choices), and need monitoring. They work best for well-defined tasks with clear success criteria.'
        ],
        bullets: [
            '<strong>Agent vs. Chat:</strong> Agents loop autonomously (perception-action-feedback) instead of single-turn responses',
            '<strong>Same Model, Bigger Loop:</strong> An agent is repeated tool use — memory lives in the context window and in external notes, not in the weights',
            '<strong>Real-World Use Cases:</strong> Coding agents (Claude Code, Codex, Cursor, Devin), research automation (paper search + summarization), customer support (knowledge base search + ticket creation), data analysis (load + clean + visualize)',
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
            { type: 'article', title: 'LangGraph: Agent Framework', meta: 'LangChain · Build production agents', url: 'https://langchain-ai.github.io/langgraph/' },
            { type: 'article', title: 'The 4 Levels of AI Agents: Workflows vs Autonomous Systems', meta: 'Duncan Anderson · Barnacle Labs', url: 'https://www.barnacle.ai/blog/2025-09-25-agents-intro' }
        ],
        image: {
            url: 'https://www.barnacle.ai/posts/2025-09-25-agents-intro/agent-loop.jpg',
            caption: 'The basic agent loop: Observe → Think → Act → Reflect',
            attribution: 'Barnacle Labs'
        }
    },
    {
        category: 'adv',
        badge: 'Conclusion',
        title: '22. Recap: Key Learnings',
        description: 'Understanding how AI models work transforms you from a passive user into an informed practitioner.',
        paragraphs: [
            'Modern AI isn\'t magic. It\'s a <strong>high-fidelity statistical mirror</strong> of human-created text, trained on trillions of tokens to predict plausible continuations and then shaped by post-training. Understanding this turns AI from a mysterious black box into an engineered system whose typical failures and capabilities make sense—even though researchers still can\'t fully explain what happens inside.',
            '<strong>Key Mental Models:</strong> (1) <strong>Frozen Artifact</strong>—chatting never teaches the model. (2) <strong>Plausibility Over Truth</strong>—it outputs what "sounds right," not what\'s correct. (3) <strong>Token-Level Reasoning</strong>—tokenization explains spelling/counting failures. (4) <strong>Context is Everything</strong>—the model sees only what is in the current prompt, and anything trimmed to fit the window is gone. (5) <strong>It\'s All Next-Token Prediction</strong>—answers, tool calls, and "thinking" are the same loop.'
        ],
        bullets: [
            '<strong>When to Use What:</strong> <strong>Prompt engineering</strong> (cheap, instant, start here) → <strong>RAG</strong> (supplies knowledge: private data, current facts, citations) → <strong>Reasoning models</strong> (complex logic/math) → <strong>Fine-tuning</strong> (changes behavior, style, or format—further training a model on your own examples; poor at adding new facts, higher cost)',
            '<strong>Model Size:</strong> An 8B model is a fast intern (runs on a laptop when compressed, prone to mistakes). A 400B+ model is a senior researcher (slower, deeper nuance). Choose the smallest model that meets your accuracy needs.',
            '<strong>Strengths:</strong> Language fluency, pattern recognition, creative generation, code completion, summarization, explaining concepts',
            '<strong>Weaknesses:</strong> Precise arithmetic, recent events, fact verification, counting/spelling — use tools (calculators, RAG, web search) to compensate'
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
            '<strong>JEPA (Joint Embedding Predictive Architecture):</strong> Yann LeCun\'s vision—predict in <em>representation space</em>, not pixels or tokens. Goal: sample-efficient learning, world models, planning. Implementations include I-JEPA (images, 2023), V-JEPA (video, 2024), and V-JEPA 2 (2025), which can plan simple robot actions.',
            '<strong>Mamba & State Space Models (SSMs):</strong> Linear or near-linear sequence complexity instead of quadratic attention. Recurrent state, long context without the same memory cost. Used in some long-context and efficient LLMs.',
            '<strong>Mixture of Experts (MoE):</strong> Sparse activation—route each token to a subset of "expert" sub-networks instead of one dense stack. Lets you scale total parameters (e.g. 400B+) while each token only uses a fraction of them. Despite the name, "experts" are not subject specialists—routing happens per token in every layer and tracks low-level patterns. Used in DeepSeek-V3, Llama-4, Gemini-2.5, Mixtral.',
            '<strong>RWKV & RetNet:</strong> RNN-like inference (constant memory per token) with parallelizable training. Pitched as "successors to the Transformer"; RWKV has an active open-source community, while RetNet remains mainly a research design.',
            '<strong>Hybrids:</strong> Models that mix attention with SSMs or linear attention, often with MoE (e.g. Jamba: attention + Mamba + MoE; Qwen3-Next: linear attention + standard attention). Combining mechanisms is a major trend.'
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