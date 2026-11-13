# Advanced ML tutorial

This repository contains small Python examples for exploring machine-learning concepts, including image compression with singular value decomposition (SVD) and dimensionality reduction with principal component analysis (PCA).

## Examples

- [Image compression and PCA example](copy_of_untitled13.py) — applies truncated SVD to color images, plots reconstruction error, and demonstrates PCA on the Iris dataset.
- [Day 1 tutorial](day1.py) — softmax, cross-entropy, and gradient checking. See the [concept notes](concept-notes.md#day1py--softmax-cross-entropy-and-gradient-checking).
- [Gradient examples](imp.py) — analytical versus numerical gradients. See the [concept notes](concept-notes.md#imppy--analytical-versus-numerical-gradients).
- [Backpropagation example](back.py) — forward propagation and backpropagation. See the [concept notes](concept-notes.md#backpy--forward-propagation-and-backpropagation).

## Requirements

The examples use Python with NumPy, Matplotlib, Pillow, and scikit-learn. Install the packages with:

```bash
python -m pip install numpy matplotlib pillow scikit-learn
```

The image-compression script loads an image from the internet in one of its examples, so an internet connection is needed unless the code is adjusted to use a local image. The script was exported from a notebook and includes a notebook-style package-install line; if running it as a regular `.py` file, remove that line after installing the requirements above.
