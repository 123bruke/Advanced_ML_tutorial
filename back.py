
import numpy as np


# ==================================
# 1. FORWARD PROPAGATION
# ==================================

def forward_pass(X, Y, params):

    W1 = params["W1"]
    b1 = params["b1"]

    W2 = params["W2"]
    b2 = params["b2"]

    N = X.shape[0]

    # First linear layer
    Z1 = X @ W1 + b1

    # ReLU activation
    A1 = np.maximum(0, Z1)

    # Second linear layer
    Z2 = A1 @ W2 + b2

    # Numerically stable softmax
    shifted = Z2 - np.max(
        Z2, axis=1, keepdims=True
    )

    log_probs = shifted - np.log(
        np.sum(np.exp(shifted), axis=1, keepdims=True)
    )

    P = np.exp(log_probs)

    # Mean cross-entropy loss
    loss = -np.sum(Y * log_probs) / N

    # Save values needed for backpropagation
    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "P": P
    }

    return loss, cache


# ==================================
# 2. MANUAL BACKWARD PROPAGATION
# ==================================

def backward_pass(Y, params, cache):

    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    P = cache["P"]

    W2 = params["W2"]

    N = X.shape[0]

    # Output gradient
    dZ2 = (P - Y) / N

    # Second layer gradients
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    # Move backward through second layer
    dA1 = dZ2 @ W2.T

    # Backpropagate through ReLU
    dZ1 = dA1 * (Z1 > 0)

    # First layer gradients
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2
    }


# ==================================
# 3. NUMERICAL GRADIENT CHECK
# ==================================

def numerical_gradients(
    X, Y, params, epsilon=1e-5
):

    numerical = {}

    for name, parameter in params.items():

        grad = np.zeros_like(parameter)

        for index in np.ndindex(parameter.shape):

            original = parameter[index]

            # Increase parameter
            parameter[index] = original + epsilon
            loss_plus, _ = forward_pass(
                X, Y, params
            )

            # Decrease parameter
            parameter[index] = original - epsilon
            loss_minus, _ = forward_pass(
                X, Y, params
            )

            # Restore original parameter
            parameter[index] = original

            # Central finite difference
            grad[index] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

        numerical[name] = grad

    return numerical


# ==================================
# 4. CREATE SAMPLE DATA
# ==================================

rng = np.random.default_rng(7)

N = 4
D = 3
H = 5
C = 3

X = rng.normal(size=(N, D))

labels = np.array([0, 2, 1, 2])

# Convert class labels to one-hot encoding
Y = np.eye(C)[labels]


# ==================================
# 5. INITIALIZE PARAMETERS
# ==================================

params = {

    "W1": rng.normal(
        scale=0.2, size=(D, H)
    ),

    "b1": np.zeros(H),

    "W2": rng.normal(
        scale=0.2, size=(H, C)
    ),

    "b2": np.zeros(C)
}


# ==================================
# 6. RUN FORWARD AND BACKWARD
# ==================================

loss, cache = forward_pass(
    X, Y, params
)

analytical = backward_pass(
    Y, params, cache
)


# ==================================
# 7. VERIFY GRADIENTS
# ==================================

numerical = numerical_gradients(
    X, Y, params
)

print("Loss:", loss)

for name in params:

    difference = np.max(
        np.abs(
            analytical[name] -
            numerical[name]
        )
    )

    print(name, "maximum gradient difference:", difference)

    assert np.allclose(
        analytical[name],
        numerical[name],
        rtol=1e-5,
        atol=1e-7
    )

print("All gradient checks passed!")
