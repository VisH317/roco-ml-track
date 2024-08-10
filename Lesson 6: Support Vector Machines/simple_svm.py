import torch

# some parameters
NUMBER_OF_FEATURES = 30

# The parameters of our SVM
parameters = torch.nn.Parameter(torch.rand(NUMBER_OF_FEATURES, 1))

example_input = torch.rand(NUMBER_OF_FEATURES)

output_prediction = parameters @ example_input # a simple matrix projection

# we can train this matrix projector to learn features over time
# using loss calculation and backpropagation

gradients = output_prediction.grad








