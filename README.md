# Neural Networks: Zero to Hero
This is a repository that contains all the files and notebooks as I followed along in Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) on YouTube.

## Micrograd
- We can represent mathematical functions as **computation graphs**. 
    - A computational graph is a directed acyclic graph (DAG).
    - Each node represents some variable.
        - The leaf nodes *(a.k.a. source nodes)* represent the inputs to the function.
        - The root node(s) *(a.k.a. sink nodes)* represents the output(s).
        - The intermediate nodes represent intermediate steps of the function.
    - Each edge represents some mathematical operation (e.g sum, product, exponentiation, etc.).
- **Backpropagation** is a method to compute the derivatives of parameters w.r.t. a functions output. To do this, we step backwards in the computation graph (using topoligical sort) and accumulate the gradients according to the **chain rule** from calculus.
- **Neural networks** are just mathematical functions where the output is a prediction. We can compose a neural network with a loss function to calculate the **loss**.
- To train a neural network we must compute the **derivatives of all the learnable parameters** w.r.t. the loss, this can be done with backpropagation.
- **Autograd** is a backpropagation engine used by PyTorch to build computational graphs and compute the derivatives of parameters w.r.t. the output (usually a loss function). It is optimized to work with tensors and take advantage of parallel computing (using GPUs).
- **Note:** Backpropagation is more general and can apply to any mathematical function. It just happens to be very useful for training neural networks.
- **Micrograd** is a simplified version of autograd at the scalar level for eductation purposes. The API of the gradient engine and the modele class is very similar to the PyTorch API.
- **Most common deep-learning mistakes:**
    1. You didn't try to overfit a single batch first. 
    2. You forgot to toggle train/eval mode for the net. 
    3. You forgot to `.zero_grad()` (in pytorch) before `.backward()`. 
    4. You passed softmaxed outputs to a loss that expects raw logits.
    5. You didn't use `bias=False` for your Linear/Conv2d layer when using BatchNorm, or conversely forget to include it for the output layer.
    6. Thinking `view()` and `permute()` are the same thing (& incorrectly using view)
- See also: [Defining new autograd functions in PyTorch](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html#pytorch-defining-new-autograd-functions)

## Ngrams vs the Neural Network Approach
- ### What is makemore?
    - Makemore makes more of things that you give it. It take one text file as input where each newline contains a new training thing.
    - Under the hood, it is an **autoregressive character-level language model**, with a wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT).
- ### Bigrams and Ngrams:
    - A bigram is a pair of consecutive written units (characters, words, tokens, etc.)
    - A bigram language model counts up all the bigrams in the training data, and then uses these counts to create **probability distributions** for the next character given the current character.
    - An Ngram is just like a bigram but with N previous characters. An Ngram language model works just like a bigram language model but takes a longer character history as the input.
    - The Ngram language model is **not scalable** to longer inputs since it requires us to store exponentially more Ngram counts as N increases.
- ### Neural network:
    - Instead of using the bigram counts to predict the next character, we can use a neural network.
    - The main advantage of using neural networks over Ngrams is that they will be able to **scale better with longer inputs**.
    - The neural network is trained using **average negative log likelihood loss** and **gradient descent**:
        - [Maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) is a method of estimating the parameters of an assumed probability distribution, given some observed data. 
        - This is achieved by maximizing a **likelihood function** w.r.t. the model parameters so that, under the assumed statistical model, the observed data is most probable.
        - The logic of maximum likelihood is both intuitive and flexible, and as such the method has become a dominant means of statistical inference.
        - Maximize the likelihood $\iff$ Maximize the log likelihood $\iff$ Minimize the negative log likelihood $\iff$ Minimize the average negative log likelihood (loss)
        - This minimization is done by gradient descent in the case of a neural network.
    - The input of the neural network is a **one-hot-encoding** of the characters.
    - The output of the neural network is **logits** (i.e. log-counts or unnormalized log-probabilities). We can use the **softmax function** to convert a vector of logits into a **probability distribution**.
    - The model can be smoothened (output probability distributions are more uniform), by adding **regularization** to the loss function.
        - Regularization can be thought of as adding **rubber bands** to the weights that pull on them as they move away from zero. This resists too much optimization w.r.t. the original loss function that could result in overfitting.
        - The same effect can be achieved with the Ngram model by padding the counts by some value to make them more unifiorm.

## MLP Model
Similar to [A Neural Probabilistic Langauge Model](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa3EtY1k1OUx5M01YTjNZLXFpQnhmd1VZS3kyZ3xBQ3Jtc0trV0s0b0lKVnIyUllyQmF6bV8zZkV0WExHZ01OWnZaSGM3SmJwc25TTEJNOHJlaVJxeTh0cXI4WGNVaEFGTC1oZU9YS0tSR3lVbm83RHo5WWdCay1iNjlSeHF2dk9XNVd5X2pNam9pVnRkd0syZmU2OA&q=https%3A%2F%2Fwww.jmlr.org%2Fpapers%2Fvolume3%2Fbengio03a%2Fbengio03a.pdf&v=TCH_1BHY58I) by Bengio et al.
- ### Network architechture
    - The network architechture is identical to the original paper, just with numbers.
    - Each of the `27` characters will be embedded in an `n` dimensional space.
    - The network takes in `k` previous chacters as the context and learns the distribution of the next character.
    - The encodings (each of length `n`) of the `k` previous words are concatenated and fed to a hiddden layer with `m` outputs.
    - The `m` outputs of the hidden layer are passed to a linear layer which produces `27` logits.
        - Pass the logits into a softmax layer to get the probability distributions for prediction.
        - Use the logits and the targets to compute the cross entropy loss for training.
- ### Implementation tips:
    - Matrix multiplication with one-hot-encoding is the same as indexing.
    - PyTorch indexing is very flexible.
    - Unbinding and concatinating can be done more efficiently with view (see this [pytorch internals blog post](http://blog.ezyang.com/2019/05/pytorch-internals/) for more details).
    - View and reshape are very similar (reshape calls view when possible).
    - Be careful when broadcasting (allign the dimensions from the right and check if its doing what you want it to do).
    - Using the built-in pytorch cross-entropy loss function is computationally faster and more numerically stable.
- ### Minibatches and stochastic gradient descent
    - Minibatches speed up training since it reduces the number of computations for each training step. 
    - However, the gradient direction for a minibatch may be different from the gradient direction of the complete training-set. 
    - We can update the parameters according to the gradient of the minibatch and this should be similar to the gradient of the entire training set on average. This is known as **stochastic gradient descent**.
- ### How to find a reasonable learning rate?
    - **Step 1:** Find a reasonable search range.
        - Try a bunch of learning rates (at different orders of magnitude) and find a reasonable search range.
            - The lower bound should be something that decreases the loss but not too quickly.
            - The upper bound should be something that causes the loss to be unstable or even explode.
    - **Step 2:** Iterate through the network with exponentially increasing learning rates from within the search range and keep track of the loss. 
        - Graph the learning rate vs the loss and see where it starts to become unstable. 
        - A good learning rate is just before that.
    - **Tip:** Lower the learning rate by 10x during the late stages of training, this is known as **learning rate decay**.
- ### Train-dev-test split
    - Neural network models can get very large with many parameters. As the capacity of the network grows, it becomes more an more capable of **overfitting** (i.e. memorising the dataset).
    - The solution is to split up the data into the training split, dev/validation split, and the test split
        - The **training split** is used to **optimize the parameters** of the model.
        - The **dev/validation** split is used for **development over the hyperparameters** (e.g size of the embedding, strength of regularization, etc.).
        - The **test split** is used to **evaluate the performance** (therefore you should only evaluate the loss on the test set sparingly).
    - A common ratio for splitting the data `80:10:10`.

## MLP Model Improvements
- ### Weight initializations
    - Appropriate weight initializations are important to ensure:
        1. The **initial loss** is reasonable 
            - We don't want a *hockey stick loss* graph i.e. the model should not start off confidently wrong. 
            - An expected initial loss can be calculated acccording to the loss function and the problem set-up. 
            - This can be done by reducing the weights of the output layer (or decreasing the gain in the case of an output layer with BatchNorm) to make the model less confident at the start.
        1. The **preactivations** of all the layers are reasonable
            - We don't want *vanishing gradients* in backpropagation i.e. the activations are too saturated.
            - We also don't want *exploding gradiens* which can be caused by vanishing activations in the network?
            - These can be avoided by appropriately initializing the weights according to the fan-in and the gain. See [Kaiming initializations](https://arxiv.org/abs/1502.01852) and the [PyTorch initialization module](https://pytorch.org/docs/stable/nn.init.html).
            - **TL;DR:** Initialize the weights by multiplying normally/unoformly distributed weights with $\frac{\text{gain}}{\sqrt{\text{fan in}}}$.
    - Vanishing gradients bcome a much bigger problem with deeper networks making appropriate weight initializations all the more important.
    - A **dead neuron** is one that always fires (activates 1 or -1 in the case of tanh) regardless of the input data, preventing it from learning anything (since gradient always vanishes).
- ### Batch Normalization
    - We want the preactivations to be roughly unit gaussian (mean 0 and standard deviation 1), not just at initialization but throughout, to enable faster and more stable training.
    - One way to achieve this is with **BatchNorm**, a layer that literally **normalizes** the preactivations and then **scales** and **shifts** them by two additional **learnable parameters** - gain and bias.
    - BatchNorm is widely adopted in deep learning but its exact reasons for success are still poorly understood.
    - The operations of BatchNorm are **differentiable** and so it just works with backpropagation just like any other layer.
    - Normally, the output of a neural network is completely dependant on the training example. But with BatchNorm, the output of the network - even for just one example - also depends on the minibatch it comes in.
    - This coupling of training examples in a batch results in an added **regularization** side effect since it acts like a sort-of data augmentation. However, this effect is usually **undesirable** as coupling batches together in this way can introduce unintended bugs.
    - **Implementation details**
        - Since BatchNorm has its own bias, the linear layer does not need another set of bias parameters (only weights).
        - A typical structure is: linear weights $\rightarrow$ BatchNorm $\rightarrow$ activation
        - If you use BatchNorm, use a lage batch size (the largest possible according to your systems resources).
        - At test time, we need to normalize the preactivations even if we are making a single prediction. This can be done by either:
            1. Looping through the entire training set at the end to compute the mean and standard deviation of all the preactivations over all training examples.
            2. Keeping track of a rolling mean and standard deviation throughout training. For smaller batches use a smaller momentum (weight of the rolling average).
    - If you can't use large batches, or are running into bugs because of batchnorm, try **linear-norm**, **instance-norm**, or **group-norm**.
- ### Diagnostic Plots
    - Understanding the relationship between weight initializations, gradient updates, learning rates, etc. can be tricky. Infact it is still an active area of research. 
    - That being said, we can peak under the hood at certain **distributions** or plot certain **values over time**, to determine if training is going well. 
    - The following are some of these diagnostic plots:
        - Distribution of activations: We want a similar saturation throughout the network (atleast at initialization).
        - Distribution of gradients of activations: We want the gradients to have the same distribution throughout the network (atleast at initialization).
        - Distribution of weights/biases
        - Distribution of gradients of weights/biases
        - Standard deviation of weights/biases
        - Standard deviation of gradients of weights/biases
        - `gradient:data` ratio: This is the ratio of standard deviation of gradients of weights to standard deviation of weights.
        - `update:data` ratio: This is the ratio of standard deviation of gradients of weights times learning rate to standard deviation of weights. We want a **update:data** ratio of about `0.001`. Use this ratio to find an appropriate learning rate.

## Backpropagation Ninja
- Why should we understand backpropagation? See [this article](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b). 
- TL;DR: Vanishing gradients, dead neurons, exploding gradients.
- Tips for implementing backpropagation manually:
    - Addition: `dsomething = 1 * dlocal_output = dlocal_output` (think of this as routing the derivative through).
    - Multiplication: `dsomething = otherthing * dlocal_output` (think of these as each thing controls the strength of the other things derivative).
    - In general: `dsomething = local_derivative * dlocal_output` (`local_derivative` is *usually* in terms of `something`)
    - The shape of `something` and `dsomething` are the same, use this property to figure out how to backpropagate based on the inputs and the chain rule.
        - There is only one way to get the right dimension for backpropagating through `@`, and it involves a `.T` operation.
        - When broadcasting is happening, there will be a `sum` operation in the backpropagation (located after multiplying by `dlocal_output`).
    - It is often useful to **write out the shapes of the inputs and outputs** when backpropagating through a segment of the computational graph.
- **Geometric interpretation of `dlogits`:** The gradient represents a pulling force of where we want the weights to go. The force sums to zero accross each row.
- See [this page](https://cs231n.github.io/optimization-2/) to read more. Here is a useful snippet:
    - The add gate always takes the gradient on its output and distributes it equally to all of its inputs, regardless of what their values were during the forward pass. This follows from the fact that the local gradient for the add operation is simply +1.0, so the gradients on all inputs will exactly equal the gradients on the output because it will be multiplied by x1.0 (and remain unchanged).
    - The max gate routes the gradient. Unlike the add gate which distributed the gradient unchanged to all its inputs, the max gate distributes the gradient (unchanged) to exactly one of its inputs (the input that had the highest value during the forward pass). This is because the local gradient for a max gate is 1.0 for the highest value, and 0.0 for all other values.
    - The multiply gate is a little less easy to interpret. Its local gradients are the input values (except switched), and this is multiplied by the gradient on its output during the chain rule. 
    - *Unintuitive effects and their consequences.* Notice that if one of the inputs to the multiply gate is very small and the other is very big, then the multiply gate will do something slightly unintuitive: it will assign a relatively huge gradient to the small input and a tiny gradient to the large input. Note that in linear classifiers where the weights are dot producted $w^T x_i$ (multiplied) with the inputs, this implies that the scale of the data has an effect on the magnitude of the gradient for the weights. For example, if you multiplied all input data examples $x_i$ by 1000 during preprocessing, then the gradient on the weights will be 1000 times larger, and you’d have to lower the learning rate by that factor to compensate. This is why preprocessing matters a lot, sometimes in subtle ways! And having intuitive understanding for how the gradients flow can help you debug some of these cases.

## WaveNet
- ### Average loss:
  - Instead of plotting the loss directly, it is often more useful to plot the average loss over multiple steps since this eliminates the noise due to minibatches. 
  - This can be achieved as follows: `plt.plot(torch.tensor(lossi).view(-1, 1000).mean(dim=1));`.
- ### PyTorch-ification:
  - The modules that we have implemented are very similar to `torch.nn` which is a neural networks API that is built ontop of `torch.tensor`.
  - Each layer can be wrapped up in a PyTorch style `nn.Module` container with the following methods: `__init__`, `__call__`, and `parameters`.
  - The layers can be stacked with a Pytorch style `nn.Sequential` container, which sequentially calls a list of layers. This allows us to wrap the whole network in a single container.
- ### Matrix multiplication of higher order tensors:
  - The matrix multiplication operation in PyTorch is quite powerful, and works with higher order tensors as well (not just matrices). The matrix multiplication acts only on the last dimension (and all the other dimensions remain unchanged). This is effectively like **adding more batch dimensions**.
  - E.g `(4, 5, 80) @ (80, 200) + (200,) = (4, 5, 200)`; where `(4, 5, 80)` is the input to the layer, `(80, 200)` are the weights, `(200,)` is the bias, and `(4, 5, 80)` is the output of the layer. In this example, the dimensions at index 0 and 1 are treated as batch dimensions, and the matrix multiplication acts on dimension 2.
- ### Original WaveNet paper:
  - The original WaveNet paper uses **convolutions over a single training sequence** to make training more **efficient** (in our case this is a single name from the training set). The network is convolved with the training sequence so that the overlapping outputs of the intermediate layers can be reused and the GPU can perform calculations in paralell. 
  - See *figure 3* from the [original wavenet paper](https://arxiv.org/pdf/1609.03499.pdf) (this is what we implemented).
  - What we have implemented is a heirachical neural network.
- ### Deep learning development process:
  - The development process of neural networks usually involves lots of reading of the PyTorch documentation, especiallly about dimension sizes and implementation details. Note that the documentation is is sometimes inaccurate.
  - There's a lot of stuff to do with shapes of tensors. It is often helpful to develop in a jupyter notebook, and then move the code over to python files to do experimentation.
  - Normally you look at the training and validation loss together and optimize on hyperparameters. This involves hyperparameter searches, python scripts with lots of arguments, lots of running experiments, looking at lots of plots, etc. all to see what works well and what doesn't.
  - In other words you would be working on the **population level** and this is what's referred to as a **training harness**. Setting this up and making it work well is a whole other topic.

## LSTM
- ### RNNs are more flexible
  - Vanilla Neural Networks (and even Convolutional Networks) are limited.
    - They can only accept a fixed-sized vector as input.
    - They can only produce a fixed-sized vector as output.
    - This mapping is done with a fixed number of computational steps.
  - On the other hand, Recurrent Neural Networks (in their most general form) are able to accept **variable length inputs**, produce **variable length outputs**, and even **vary the number of computational steps** for different lengths.
- ### Classification of RNNs
  - RNNs can be classified into **5 types** depending on the input and output lengths (see the image in the reference blogpost for a visual explanation):
    - **One-to-one:** this is just a Vanilla Neural Network.
    - **One-to-many:** e.g. image captining (fixed size image to variable size caption).
    - **Many-to-one:** e.g sentiment analysis (variable sized text to binary sentiment).
    - **Many-to-many (asynced):** e.g machine translation (variable sized input sequence and output sequence).
    - **Many-to-many (synced):** e.g video classification (variable sized input sequence, output sequence is same size as input sequence).
  - Note that a sequence refers to a sequence of vectors. The sequence has variable length but the size of each vector in the sequence is fixed.
- ### Optimizing over functions
  - The ability for RNNs to **vary the number of computational steps (while maintaining the same number of learnable parameters)** is where the magic happens.
  - **RNNs are Turing-Complete** (but similar to universal approximation theorems for neural nets you shouldn’t read too much into this. In fact, forget I said anything.)
  - If training vanilla neural nets is optimization over functions, training recurrent nets is optimization over programs.
- ### Sequential processing in absence of sequences
  - You might be thinking that having sequences as inputs or outputs could be relatively rare.
  - However, even if your inputs/outputs are fixed vectors, it is still possible to use this powerful formalism to **process them in a sequential manner**.
  - You’re learning stateful programs that process your fixed-sized data.

## Transformer
- ### Tokenizer
    - Tokenizers convert between text data and integer encodings by dividing the text up into tokens.
    - Here we used a character level tokenizer (each character is a unique integer), however in practice people usually use **sub-word level tokenizers**.
    - They have **larger vocabulary sizes**, but use **fewer numbers** to represent a given text.
    - There is a **tradeoff** between vocabulary size and representation size. 
    - Popular **sub-word tokenizers** are [sentencepiece](https://github.com/google/sentencepiece) and [tiktoken](https://github.com/openai/tiktoken).
- ### Batches
    - The model is trained in minibatches of encoded sequences of text.
    - Each batch has 2 dimensions. The first dimension is the *batch* dimension, and the second dimension is the *time* dimension.
    - The transformer learns to predict the target given context of any length up to `block_size`. 
    - This is not just for efficiency, and is infact necessery for inference since the transformer will need to be able to generate text from scratch (with no context, then context of length 1, and so on...).
- ### Averaging with matrices
    - The notebook illustrates a mathematical trick where we can "average" using matrix multiplication instead of using for loops. 
    - This can also be extended to weighted averages very easily. Here the weights are normalized to sum to one and not look into the future of the time dimension.
    - *Version 3* is especially useful since it startes with "affinities" between different tokens, and they can be data dependant instead of just being set to zero.
    <br>`affinities = torch.zeros((T, T)) # affinities between tokens `
    <br> `tril = torch.tril(torch.ones(T, T))`
    <br> `weights = affinities.masked_fill(tril==0, float('-inf')) # prevent lookahead`
    <br> `weights = F.softmax(weights, dim=1)as before`
    <br> `xbow3 = weights @ x`
- ### Self-attention
    - #### Crux of self-attention
        - Every token will emit three vectors - **query**, **key**, and **value**
        - The key tells other tokens what it *has*, and the query tells other tokens what it's *looking for*.
        - The value contains the *information* of the token that is relevant for other tokens.
        - The **dot product** between a key and query for a token pair become the *affinities* between tokens i.e. how interested is one token with another.
        - For example, a pronoun could be looking for a noun, so the dot product between the query of the pronoun and the key of the noun could be high.
        - The model then computes the weight matrix by performing softmax on the masked affinities.
        - This weight matrix is then multiplied with the values of all the tokens to produce the output of the self-attention layer.
    - #### Additional notes
        - Attention is a **communication mechanism**. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.
        - There is no notion of space. Attention simply acts over a set of vectors. This is why we need to **positionally encode tokens**.
        - Each example across batch dimension is of course processed completely independently and never "talk" to each other
        - In an **encoder** attention block just delete the single line that does masking with `tril`, allowing all tokens to communicate. This block here is called a **decoder** attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.
        - **Scaled attention** additionally divides the `weights` by 1/sqrt(head_size). This makes it so when input `k`, `q` are unit variance, `weights` will be unit variance too and Softmax will stay diffuse and not saturate too much. This prevents Softmax from converging to one-hot vectors.
- ### Additional optimizations
    - #### **Multi-head attention**
        - Often, the tokens often "have a lot to talk about". This communication can be facillitated by multiple attention heads between the same tokens.
        - For example one of the heads could be looking for consonants, while the other could be looking for vowels, etc.
        - The final output is a concatination between all the attention heads.
    - #### **Communication and computation**
        - It is often useful to let the network **"think over"** the outputs of the attention blocks. This can be achieved by adding more **computation** steps using feed-forward layers. 
        - Infact a common architechure is to **intersperse** the attention blocks, that allow communication between tokens, with feed-forward blocks, that allow tokens to individually process the results of this communication.
    - #### **Skip/residual connections**
        - Deep neural networks often suffer from optimization issues due to vanishing gradients. Skip connections (a.k.a. residual connections) help with this.
        - The main concept is that you "branch" the data off to transform the data in some way, but also add the non-transformed version back to the "main branch". 
        - Since addition routes gradients, this creates a "gradient super-highway" that allows gradients to backpropagate more effectively to the earlier layers. 
        - The "branches" off the highway are where the computation happens, and they are initialized in such a way that they barely contribute any gradients, but eventually they become more relevant as training goes on.
    - #### **LayerNorm**
        - This is another innovation that helps with numerical stability of the gradients during optimization of deep neural networks.
        - The idea is the same as BatchNorm, we want every neuron to have unit gaussian outputs. 
        - However instead of normalizing over the batch dimension, we normalize over the time dimension. 
        - This is better than BatchNorm since batches are no longer coupled together and we don't need to keep track of training vs testing.
        - The transformer architechture has stood the test of time and has remained relatively unchanged for quite some time since it's introduction in 2017. One change that was made was the pre-norm formulation that adds a LayerNorm before the transformer block, and another one before the feed-forward block.
    - #### **Dropout**
        - This is a regularization technique that helps prevent overfitting.
        - Every forward pass, a random subset of neurons are disabled, so only a subset of the network is trained in each optimization step.
        - This has the effect of training an ensemble of networks that are merged together at test time.
- ### Cross-attention
    - Self-attention just means that the keys and values are produced from the same source as queries. In **cross-attention**, the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)
    - When we want the generated text to be conditioned on some previous text, we can use an encoder transformer blocks (without the traingular mask), to generate the keys and values, and a decoder transformer block to generate the queries.
    - These can be combined to form an encoder-decoder transformer. See the [attention is all you need](https://arxiv.org/abs/1706.03762) paper for more details.
- ### How to train ChatGPT
    - #### Stage 1: **Pretraining**
        - The first stage is to train a decoder transformer langauge model on all the data from the internet that simply learns to babble on
        - Here we trained a character level model with 10 million parameters on a dataset with 1 million character level tokens tokens which would be about 300,000 tokens in the OpenAI vocabulary.
        - The biggest OpenAI GPT 3 transformer has 175 billion parameters on 300 billion tokens. These numbers are not even large compared to current standards. See OpenAI's [GPT-3 paper](https://arxiv.org/abs/2005.14165) for more details.
        - Due to scale, this becomes a massive infrastructure challenge.
    - #### Stage 2: **Fine-tuning**
        - So far the pre-trained model will just autocomplete and babble pretty much anything. It could complete the sentence, it could write a news article, it could ask more questions, pretty much anything that it sees on the internet.
        - The next stage is to "align it" to be an assistant. See this [blog post](https://openai.com/blog/chatgpt) by OpenAI for more information.
        - Step 1 fine-tunes this model based on a small dataset of good example "question-answer" pairs. This works because pre-trained LLMs are very sample efficient.
        - Step 2 trains a reward model by asking humans to rank different responses from bad to good. This reward model is a way to score any response.
        - Step 3 uses the PPO a reinforcement learning algorithm to further fine-tune the model based on the reward model.