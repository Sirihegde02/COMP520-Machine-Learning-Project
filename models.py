import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        score = nn.as_scalar(self.run(x))
        return 1 if score >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        while True:
            mistakes = 0
            for x, y in dataset.iterate_once(1):
                y_scalar = nn.as_scalar(y)
                pred = self.get_prediction(x)
                if pred != y_scalar:
                    self.w.update(x, y_scalar)
                    mistakes += 1
            if mistakes == 0:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        self.hidden_size1 = 512
        self.hidden_size2 = 256
        self.batch_size = 200
        self.learning_rate = 0.05

        self.w1 = nn.Parameter(1, self.hidden_size1)
        self.b1 = nn.Parameter(1, self.hidden_size1)
        self.w2 = nn.Parameter(self.hidden_size1, self.hidden_size2)
        self.b2 = nn.Parameter(1, self.hidden_size2)
        self.w3 = nn.Parameter(self.hidden_size2, 1)
        self.b3 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        h1_preact = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        h1 = nn.ReLU(h1_preact)
        h2_preact = nn.AddBias(nn.Linear(h1, self.w2), self.b2)
        h2 = nn.ReLU(h2_preact)
        out = nn.AddBias(nn.Linear(h2, self.w3), self.b3)
        return out

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        pred = self.run(x)
        return nn.SquareLoss(pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        max_epochs = 5000
        target_loss = 0.0195
        epoch = 0
        while epoch < max_epochs:
            total_loss = 0.0
            batches = 0
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                self.w1.update(grads[0], -self.learning_rate)
                self.b1.update(grads[1], -self.learning_rate)
                self.w2.update(grads[2], -self.learning_rate)
                self.b2.update(grads[3], -self.learning_rate)
                self.w3.update(grads[4], -self.learning_rate)
                self.b3.update(grads[5], -self.learning_rate)

                total_loss += nn.as_scalar(loss)
                batches += 1
            avg_loss = total_loss / max(1, batches)
            if avg_loss <= target_loss:
                break
            epoch += 1

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.hidden_size = 200
        self.batch_size = 100
        self.learning_rate = 0.5
        
        # First layer: input (784) -> hidden (200)
        self.w1 = nn.Parameter(784, self.hidden_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        
        # Second layer: hidden (200) -> output (10)
        self.w2 = nn.Parameter(self.hidden_size, 10)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        # First layer: Linear transformation + bias + ReLU
        h1_preact = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        h1 = nn.ReLU(h1_preact)
        
        # Second layer: Linear transformation + bias (NO ReLU at the end)
        out = nn.AddBias(nn.Linear(h1, self.w2), self.b2)
        return out

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        pred = self.run(x)
        return nn.SoftmaxLoss(pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        max_epochs = 50
        target_accuracy = 0.975  # 97.5% validation accuracy threshold
        
        for epoch in range(max_epochs):
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                grads = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
                
                self.w1.update(grads[0], -self.learning_rate)
                self.b1.update(grads[1], -self.learning_rate)
                self.w2.update(grads[2], -self.learning_rate)
                self.b2.update(grads[3], -self.learning_rate)
            
            # Check validation accuracy
            val_accuracy = dataset.get_validation_accuracy()
            if val_accuracy >= target_accuracy:
                break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
