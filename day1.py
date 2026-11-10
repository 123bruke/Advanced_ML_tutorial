
import numpy as np

def softmax(z):
    # Numerical stability
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


def cross_entropy(z, y):
    p = softmax(z)
    return -np.sum(y * np.log(p))


# Example logits
z = np.array([2.0, 1.0, 0.1], dtype=np.float64)

# Correct class is Dog
y = np.array([0.0, 1.0, 0.0])

# Analytical gradient
p = softmax(z)
analytical = p - y

# Numerical gradient
epsilon = 1e-5
numerical = np.zeros_like(z)

for i in range(len(z)):
    z_plus = z.copy()
    z_minus = z.copy()

    z_plus[i] += epsilon
    z_minus[i] -= epsilon

    loss_plus = cross_entropy(z_plus, y)
    loss_minus = cross_entropy(z_minus, y)

    numerical[i] = (
        loss_plus - loss_minus
    ) / (2 * epsilon)


print("Analytical gradient:", analytical)
print("Numerical gradient: ", numerical)

print("Maximum difference:",
      np.max(np.abs(analytical - numerical)))

print("Gradients match:",
      np.allclose(analytical, numerical, atol=1e-7))
