# Concept notes

These notes explain the machine-learning ideas illustrated by the Python scripts. The links at the top of each section open the corresponding source file.

## [`day1.py`](https://github.com/123bruke/Advanced_ML_tutorial/blob/main/day1.py) — softmax, cross-entropy, and gradient checking

- **Logits:** `z` contains the model's unnormalized scores for each class. Larger scores indicate classes the model favors, but the scores are not probabilities yet.
- **Softmax:** subtracting the largest logit before exponentiation improves numerical stability without changing the resulting probabilities. Exponentiating and normalizing makes the values sum to one.
- **One-hot labels:** `y` represents the correct class with a 1 at its index and 0 elsewhere. Here the second class (Dog) is the target.
- **Cross-entropy:** the loss `-sum(y * log(p))` is the negative log-probability assigned to the correct class. A confident correct prediction has a low loss; a confident incorrect one has a high loss.
- **Analytical gradient:** for softmax followed by cross-entropy, the derivative of the loss with respect to each logit is `p - y`.
- **Numerical gradient check:** the script estimates each derivative with a centered finite difference: `[L(z + ε) - L(z - ε)] / (2ε)`. Comparing this approximation with the analytical derivative is a practical way to catch calculus or implementation mistakes. The small `epsilon` balances truncation error against floating-point rounding.

## [`imp.py`](https://github.com/123bruke/Advanced_ML_tutorial/blob/main/imp.py) — analytical versus numerical gradients

This script repeats the softmax-cross-entropy gradient check with a different example and class label. It demonstrates that the `p - y` result is not specific to one input: the analytical gradient should agree with the centered finite-difference estimate for every logit. `np.allclose` allows a small tolerance because numerical derivatives are approximate and floating-point arithmetic is not exact.

## [`back.py`](https://github.com/123bruke/Advanced_ML_tutorial/blob/main/back.py) — forward propagation and backpropagation

- **Network structure:** the example is a two-layer fully connected classifier: input → hidden layer → class scores. `W1`, `b1`, `W2`, and `b2` are its trainable weights and biases.
- **Forward propagation:** matrix multiplication and bias addition produce `Z1`; ReLU (`max(0, Z1)`) creates hidden activations `A1`; the next affine layer produces class logits `Z2`.
- **Stable softmax and loss:** subtracting each row's maximum logit prevents unnecessarily large exponentials. The code computes log-probabilities directly and averages cross-entropy across the `N` examples. `Y` is one-hot encoded, with one row per example.
- **Backpropagation:** the output-logit derivative is `(P - Y) / N`. The chain rule propagates it backward through the second affine layer, through ReLU (whose derivative is 1 for positive inputs and 0 otherwise), and through the first affine layer. Matrix products and sums accumulate weight and bias gradients over the batch.
- **Gradient checking:** `numerical_gradients` perturbs each parameter slightly in both directions and estimates its derivative with a centered difference. The final comparisons check every weight and bias gradient against the manually derived backpropagation result. This check is for learning and debugging; finite differences are computationally expensive and generally are not used to train real models.
- **Sample data and initialization:** a seeded random generator makes the toy example repeatable. Random inputs and small initial weights provide a compact test case; the labels are converted into one-hot rows for the loss calculation.
