'''
Autograd automatically calculates the gradients.
Turn on or off gradients altogether with torch.set_grad_enabled(True|False)
'''

import torch

torch.manual_seed(7)

# Example 1
x = torch.randn(2, 2, requires_grad=True)
y = x**2
z = y.mean()

print(y.grad_fn)  # shows the function that generated this variable. --> power operation. PowBackward0.
print(z.grad_fn)  # shows the function that generated this variable. --> mean operation. MeanBackward0.

print(x.grad)  # gives none. We didn't do any backward operation yet.

# Let's find the gradient of z with respect to x.
print(z.backward())  # z = x^2/4 z' = 2x/4 = x/2  print gives None
print(x.grad)  # after the backward operation gives gradient.
print(x/2)  # same with x.grad



print('done')