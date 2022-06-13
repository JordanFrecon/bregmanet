import torch
import torch.nn as nn
import bregmanet.utils.weight as weight
import bregmanet.utils.activation as act


class MLP(nn.Module):
    r"""
        Bregman Multilayer Perceptron

        Arguments:
            - activation_name (string): activation function.
            - input_dim (integer): size of the input data flattened into a 1d tensor.
            - hidden_dim (list of integer): number of hidden neurons per layer.
            - output_dim (integer): number of output neurons.
            - version (string): switch between 'standard' and 'bregman' MLP. (default: 'bregman')
            - init (string): switch between 'random' and 'deterministic' initialization of the weights and biases.
             (default: 'random')
            - weight_norm (boolean): if True, perform layer-wise weights normalization. (default: False)

        Example:
            model = MLP(activation_name='sigmoid', input_dim=1024, hidden_dim=[1024, 512, 1024], output_dim=10)

        """

    def __init__(self, activation, version='bregman', hidden_dim=None, input_dim=None, output_dim=1,
                 init='random', weight_norm=False):
        super().__init__()
        version = version.lower()
        activation = activation.lower()
        init_param = weight.parameter_initialization(version=version, init_type=init)
        if version == 'bregman':
            self.__class__.__name__ = 'BregmanMLP'

        # Parameters
        self.num_neurons = [input_dim] if hidden_dim is None else hidden_dim
        self.num_layers = self.num_neurons.__len__()
        self.weight_norm = weight_norm
        self.version = version

        # Hidden layers
        self.lin = nn.ModuleList()
        self.reparametrization = nn.ModuleList()
        for in_neurons, out_neurons in zip([input_dim] + self.num_neurons[:self.num_layers - 1], self.num_neurons):

            # Linear part associated to the offset
            if in_neurons == out_neurons or self.version == 'standard':
                # ... then no need to initialize
                self.reparametrization.append(nn.Identity())
            else:
                # ... then initialize with random weights on the simplex
                self.reparametrization.append(weight.linear_with_init(in_neurons, out_neurons, init='simplex'))

            # Classical linear part
            self.lin.append(weight.linear_with_init(in_neurons, out_neurons, init=init_param['hidden'],
                                                    weight_norm=weight_norm))
        self.activation, self.offset, self.range = act.get(activation_name=activation, version=version)

        # Output layer
        self.output = weight.linear_with_init(self.num_neurons[-1], output_dim, init=init_param['output'])

    def forward(self, xb):

        # Hidden layers
        for idl in range(self.num_layers):

            # Constraint on offset weights
            if self.version == 'bregman':
                self.reparametrization[idl] = weight.constraint(self.reparametrization[idl])

            # Perform forward pass
            if self.version == 'bregman':
                x_offset = torch.clamp(self.reparametrization[idl](xb), self.range[0], self.range[1])
                xb = self.activation(self.offset(x_offset) + self.lin[idl](xb))
            else:
                xb = self.activation(self.lin[idl](xb))

        # Output layer
        xb = self.output(xb)

        return xb
