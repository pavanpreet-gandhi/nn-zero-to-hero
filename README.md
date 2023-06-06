# Neural Networks: Zero to Hero
This is a repository that contains all the files and notebooks as I followed along in Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) course on YouTube.

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