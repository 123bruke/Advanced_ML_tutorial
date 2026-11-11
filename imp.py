
import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

def loss_function(z, y):
    p = softmax(z)
    return -np.sum(y * np.log(p))

# Input logits
z = np.array([2.0, 1.0, 0.0])

# True labels: Cat, Dog, Bird
y = np.array([1.0, 0.0, 0.0])

# Analytical gradient
p = softmax(z)
analytical_gradient = p - y

# Numerical gradient
epsilon = 1e-5
numerical_gradient = np.zeros_like(z)

for i in range(len(z)):
    z_plus = z.copy()
    z_minus = z.copy()

    z_plus[i] += epsilon
    z_minus[i] -= epsilon

    loss_plus = loss_function(z_plus, y)
    loss_minus = loss_function(z_minus, y)

    numerical_gradient[i] = (
        loss_plus - loss_minus
    ) / (2 * epsilon)

print("Analytical:", analytical_gradient)
print("Numerical: ", numerical_gradient)

print(
    "Gradients match:",
    np.allclose(
        analytical_gradient,
        numerical_gradient,
        atol=1e-7
    )
)
