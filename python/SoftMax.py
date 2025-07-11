import numpy as np

layer_output = [4.8, 1.21, 2.385]
print(layer_output)

exp_values = np.exp(layer_output)

print(exp_values)

normalize_values = exp_values / np.sum(exp_values)

print(normalize_values)
print(sum(normalize_values))
