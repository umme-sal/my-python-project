# Matrix and Vector Operations in Python

This repository contains two Python modules for performing a variety of operations on **matrices** and **vectors**, useful for data science, linear algebra, machine learning, and more.

---

## 📁 Modules

### `matrix.py`

Provides advanced matrix functionalities:
- `scaler_multi(matrix, k)` – Multiply matrix by a scalar
- `transpose(matrix)` – Transpose of a matrix
- `determinant(matrix)` – Calculate determinant (supports >2x2)
- `matrix_multiply(A, B)` – Matrix multiplication
- `cofactor_matrix(matrix)` – Compute matrix of cofactors
- `adjoint_matrix(matrix)` – Find adjoint of matrix
- `inverse_matrix(matrix)` – Matrix inverse (using adjoint/det)
- `row_echelon(matrix)` – Get row echelon form
- `matrix_rank(matrix)` – Rank of a matrix
- `matrix_power(matrix, n)` – Matrix exponentiation
- `lu_decomposition(matrix)` – LU decomposition
- `eigenvalues_2x2(matrix)` – Eigenvalues for 2x2 matrix
- `eigenvalues_3x3(matrix)` – Eigenvalues for 3x3 matrix

### `vector.py`

Includes a broad range of vector operations:
- Basic arithmetic: `vector_addition`, `vector_subtraction`
- Products: `dot_product`, `cross_product`, `hadamard_product`, `outer_product`
- Vector properties: `magnitude`, `norm`, `unit_vector`
- Projections: `vector_projection`, `projection_plane`
- Reflections (with 3D visualization): `plot_vector_reflection_plane`, `plot_reflection`
- Distances: Euclidean, Manhattan, Cosine
- Advanced:
  - `rotation_3D`: Rotate vector in 3D using Rodrigues’ formula
  - `gram_schmidt`: Orthogonalize a set of vectors
  - `linear_combination`, `is_in_span`, `is_linearly_dependent`
  - `convolution_1d`, `linear_interpolation`

---

## 🧪 Example Usage

Example usage snippets are included at the bottom of each module. Run the Python files directly to test functionalities.

```python
python matrix.py
python vector.py

Requirement:
pip install matplotlib

How to install the module:
pip install linear_algebra_mod
