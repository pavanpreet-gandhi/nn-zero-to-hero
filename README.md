# Neural Networks: Zero to Hero
This is a repository that contains all the files and notebooks as I followed along in Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) course on YouTube.

## Micrograd
- We can represent mathematical functions as **computation graphs**. 
    - A computational graph is a directed acyclic graphs (DAG).
    - Each node represents some variable.
        - The leaf nodes *(a.k.a. source nodes)* represent the inputs to the function.
        - The root node(s) *(a.k.a. sink nodes)* represents the output(s).
        - The intermediate nodes represnt intermediate steps of the function.
    - Each edge represents some mathematical operation (e.g sum, product, exponentiation).
- **Backpropagation** is a method to compute the derivatives of parameters w.r.t. a functions output. To do this, we step backwards in the computation graph (using topoligical sort) and accumulate the gradients according to the **chain rule** from calculus.
- **Neural networks** are just mathematical functions where the output is a prediction. We can compose a neural network with a loss function to calculate the **loss**.
- To train a neural network we must compute the **derivatives of all the learnable parameters** w.r.t. the loss, this can be done with backpropagation.
- **Autograd** is a backpropagation engine used by PyTorch to build computational graphs and compute the derivatives of the leaf nodes w.r.t. the output (usually a loss function). It is optimized to work with tensors and take advantage of parallel computing (using GPUs).
- **Note:** Backpropagation is more general and can apply to any mathematical function. It just happens to be very useful for training neural networks.
- **Micrograd** is a simplified version of autograd at the scalar level for eductation purposes. The API of the gradient engine and the neuron class is very similar to the PyTorch API.
- **Most common deep-learning mistakes:**
    1. You didn't try to overfit a single batch first. 
    2. You forgot to toggle train/eval mode for the net. 
    3. You forgot to `.zero_grad()` (in pytorch) before `.backward()`. 
    4. You passed softmaxed outputs to a loss that expects raw logits.
    5. You didn't use `bias=False` for your Linear/Conv2d layer when using BatchNorm, or conversely forget to include it for the output layer.
    6. Thinking `view()` and `permute()` are the same thing (& incorrectly using view)
- See also: [Defining new autograd functions in PyTorch](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html#pytorch-defining-new-autograd-functions)

## Makemore (part 1)
- ### What is makemore?
    - Makemore makes more of things that you give it. It take one text file as input where each newline contains a new training thing.
    - Under the hood, it is an **autoregressive character-level language model**, with a wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT).
- ### Bigrams and Ngrams:
    - A bigram is a pair of consecutive written units (characters, words, tokens, etc.)
    - A bigram language model counts up all the bigrams in the training data, and then uses these counts to create **probability distributions** for the next character given teh current character.
    - An Ngram is just like a bigram but with N previous characters. An Ngram language models works just like a bigram language model but takes a longer character history as the input.
    - The Ngram language model is **not scalable** to longer inputs since it requires us to store exponentially more Ngram counts as N increases.
- ### Neural network:
    - Instead of using the bigram counts to predict the next character, we can use a neural network.
    - The main advantage of using neural networks over Ngrams is that they will be able to **scale better with longer inputs**.
    - The neural network is trained using the **average negative log likelihood loss** and **gradient descent**:
        - [Maximum likelihood estimation](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) is a method of estimating the parameters of an assumed probability distribution, given some observed data. 
        - This is achieved by maximizing a **likelihood function** w.r.t. the model parameters so that, under the assumed statistical model, the observed data is most probable.
        - The logic of maximum likelihood is both intuitive and flexible, and as such the method has become a dominant means of statistical inference.
        - Maximize the likelihood $\iff$ Maximize the log likelihood $\iff$ Minimize the negative log likelihood $\iff$ Minimize the average negative log likelihood (loss)
    - The input of the neural network is a **one-hot-encoding** of the characters.
    - The output of the neural network is **logits** (i.e. log-counts). We can use the **softmax function** to convert a vector of logits into a **probability distribution**.
    - The model can be smoothened (output probability distributions are more uniform), by adding **regularization** to the loss function.
        - Regularization can be thought of as adding **rubber bands** to the weights that pull on them as they move away from zero. This resists too much optimization w.r.t. the original loss function that could result in overfitting.
        - The same effect can be achieved with the Ngram model by padding the counts by some value to make them more unifiorm.
