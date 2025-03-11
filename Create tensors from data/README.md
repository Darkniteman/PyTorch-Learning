# Create Tensors in PyTorch

## Overview
This project demonstrates various ways to create tensors using PyTorch, including initialization from lists, NumPy arrays, and different tensor generation methods such as zeros, ones, and random values.

## Tensor Creation Methods
### 1. Creating Tensors from Existing Data
- **From Python List:**
  ```python
  tensor_list = torch.tensor([1, 1, 1])
  ```
- **From NumPy Array:**
  ```python
  import numpy as np
  nparray = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
  tensor_array = torch.tensor(nparray)
  ```

### 2. Creating Tensors with Predefined Values
- **Zeros and Ones:**
  ```python
  tensor_zeros = torch.zeros(2, 2)
  tensor_ones = torch.ones(2, 2)
  ```

### 3. Creating Random Tensors
- **Uniform Distribution (`rand`)**:
  ```python
  random_tensor = torch.rand(3, 3)  # Values between [0,1]
  ```
- **Normal Distribution (`randn`)**:
  ```python
  normal_tensor = torch.randn(3, 3)  # Mean=0, Std=1
  ```

### Difference Between `rand` and `randn`
| Function | Distribution Type | Range |
|----------|------------------|--------|
| `rand`   | Uniform          | [0,1]  |
| `randn`  | Normal (Gaussian) | Mean=0, Std=1 (values can be negative) |

### 4. Specifying Tensor Data Types
- **Integer Tensor:**
  ```python
  tensor_int = torch.tensor([1, 1, 1], dtype=torch.int)
  ```
- **Float Tensor:**
  ```python
  tensor_float = torch.tensor([1.5, 1.7, 1.9], dtype=torch.float32)
  ```

### 5. Creating Tensors on GPU (if available)
```python
if torch.cuda.is_available():
    tensor_gpu = torch.tensor([1, 1, 1], device='cuda')
    print("Tensor on GPU:", tensor_gpu)
else:
    print("CUDA is not available. Running on CPU.")
```

## How to Run the Script
1. Install PyTorch if not already installed:
   ```bash
   pip install torch
   ```
2. Run the script:
   ```bash
   python create_tensors_from_data.py
   ```

## License
This project is open-source and free to use.

