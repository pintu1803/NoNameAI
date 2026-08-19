##19/08/2026
### Activation functions.
"""
Sigmoid
ReLU(x) -> Max{0, x} -> 0 for numbers < 0 and positive numbers are kept same.
"""

"""
1. Why do neural networks need activations? 
Activations enable the networks to model non-linear relationships.
2.  Why can't we stack only Linear layers? 
The final result of any number of Linear layers stacked is still a linear equation/layer.
3.  Why did Sigmoid dominate for 20 years? 
Sigmoid results between (0, 1) and it seems like probability.
People loved it, compared the neurons with biological neurons which fire gradually.
4.  Why did ReLU almost completely replace Sigmoid in hidden layers? 
Sigmoid breaks when we try to make network deeper.
Deeper means when we try to model non-linear (complex functional relationships).
ReLU(x) -> Max(0, x) this breaks the linearity when multiple linear layers are stacked.
5.  What is the **vanishing gradient problem**? 
As per the chain rule, gradients of layer 1 = grad of layer n * grad of layer n-1 * .. * grad of layer 2
Now, suppose each layer's grad is 0.1, then layer-1 computes the grad = (0.1)^n, which is almost 0
so, the gradient of layer vanished. This is it.
6.  Why can ReLU cause **dead neurons**? 
ReLU(x) -> Max(0,x) => let say a neuron always gives negative result, then it's ReLU(x) = 0 always.
so, logically this neuron is dead.
7.  When do we use **Sigmoid**, **Tanh**, **ReLU**, **Leaky ReLU**, and **Softmax**?
Sigmoid is used in hidden layers when we want activation results as probabilities.
It is used in binary-class classification.
ReLU is used in hidden layer when we want to model non-linear behaviour.
Leaky ReLU is used when dead neurons(dead ReLU) becomes a concern
Softmax is used in multi-class-output-layer when we want the output as probabilities s.t. sum of all is 1.

"""

"""
Inserting a non-linear function between the Linear layers break the "still just one big linear equation" collapse.
Linear -> Activation -> Linear -> Activation -> Linear  => multi-layer perception

Instead, torch provides nn.Sequential() which reduces the boilerplate code and provides same underlying mechanism.

nn.MSELoss() for regression) and nn.CrossEntropyLoss() for classification
CrossEntropyLoss expects raw logits and applies log_softmax internally ->
so we don't manually apply softmax before this loss.
"""