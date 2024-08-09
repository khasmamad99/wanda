import torch
import torch.nn as nn

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        
        
class QueryOrKeyProjectionLayerWrapper:
    """This class wraps a proj_q or a proj_k layer for specific operations."""
    def __init__(self, layer: torch.nn.Linear):
        self.layer = layer
        num_out_channels, num_in_channels = layer.weight.data.shape

        self.average_input_activations_sqrd_norm = torch.zeros((num_in_channels,), device=layer.weight.device)
        self.average_output_activations_sqrd_norm = torch.zeros((num_out_channels,), device=layer.weight.device)
        self.num_samples = 0
        
        
    def add_batch(self, input_activations: torch.Tensor, output_activations: torch.Tensor):
        if len(input_activations.shape) == 2:
            input_activations = input_activations.unsqueeze(0)
        if len(output_activations.shape) == 2:
            output_activations = output_activations.unsqueeze(0)
            
        batch_size, sequence_length, num_in_channels = input_activations.shape
        batch_size, sequence_length, num_out_channels = output_activations.shape
        
        input_activations = input_activations.type(torch.float32)
        output_activations = output_activations.type(torch.float32)
        
        input_activations = input_activations.to(self.average_input_activations_sqrd_norm.device)
        output_activations = output_activations.to(self.average_input_activations_sqrd_norm.device)
        
        if len(input_activations.shape) == 3:
            input_activations = input_activations.view(-1, num_in_channels)
        if len(output_activations.shape) == 3:
            output_activations = output_activations.view(-1, num_out_channels)
        
        sum_of_input_activations_sqrd_norms = (
            (self.average_input_activations_sqrd_norm * self.num_samples) 
            + (torch.norm(input_activations, p=2, dim=0).pow(2))
        )
        sum_of_output_activations_sqrd_norms = (
            (self.average_output_activations_sqrd_norm * self.num_samples) 
            + (torch.norm(output_activations, p=2, dim=0).pow(2))
        )
        
        self.num_samples += batch_size
        self.average_input_activations_sqrd_norm = sum_of_input_activations_sqrd_norms / self.num_samples
        self.average_output_activations_sqrd_norm = sum_of_output_activations_sqrd_norms / self.num_samples
        
                  
        
class ValueProjectionLayerWrapper:
    """This class wraps a proj_v layer for specific operations."""
    def __init__(self, layer: torch.nn.Linear):
        self.layer = layer
        num_out_channels, num_in_channels = layer.weight.data.shape

        self.input_activations = None
        self.average_input_activations_and_attention_weights_sqrd_norm = torch.zeros(
            size=(num_out_channels,), 
            device=layer.weight.device,
        )
        self.num_samples = 0
        
        
    def add_batch(self, input_activations: torch.Tensor, output_activations: torch.Tensor):
        if len(input_activations.shape) == 2:
            input_activations = input_activations.unsqueeze(0)
        self.input_activations = input_activations
        
        
    def update(self, attention_weights: torch.Tensor):
        assert self.input_activations is not None, "input_activations is None. Call add_batch() first."
        
        if len(attention_weights.shape) == 2:
            attention_weights = attention_weights.unsqueeze(0)
            
        if len(attention_weights.shape) == 3:
            attention_weights = attention_weights.unsqueeze(0)
        
        self.input_activations = self.input_activations.to(self.average_input_activations_and_attention_weights_sqrd_norm.device)
        attention_weights = attention_weights.to(self.average_input_activations_and_attention_weights_sqrd_norm.device)
        
        self.input_activations = self.input_activations.type(torch.float32)
        attention_weights = attention_weights.type(torch.float32)
        num_heads, batch_size, seq_len, seq_len = attention_weights.shape
        activations_and_attention_weights = (
            self.input_activations.transpose(1, 2)
            @ attention_weights.transpose(0, 1)
        )  # -> (num_heads, batch_size, num_in_channels, seq_len)
        
        activations_and_attention_weights = activations_and_attention_weights.transpose(2, 3)
        activations_and_attention_weights = activations_and_attention_weights.reshape(
            (num_heads * batch_size * seq_len, -1)
        )
        
        sum_of_input_activations_and_attenion_weights_sqrd_norms = (
            (self.average_input_activations_and_attention_weights_sqrd_norm * self.num_samples) 
            + (torch.norm(activations_and_attention_weights, p=2, dim=0).pow(2))
        )
        self.num_samples += batch_size
        self.average_input_activations_and_attention_weights_sqrd_norm = (
            sum_of_input_activations_and_attenion_weights_sqrd_norms / self.num_samples
        )
        
        
class FullyConnectedLayerWrapper:
    """This class wraps a fc layer for specific operations."""
    def __init__(self, layer: torch.nn.Linear):
        self.layer = layer
        num_out_channels, num_in_channels = layer.weight.data.shape

        self.average_input_activations_sqrd_norm = torch.zeros((num_in_channels,), device=layer.weight.device)
        self.num_samples = 0
        
        
    def add_batch(self, input_activations: torch.Tensor, output_activations: torch.Tensor):
        if len(input_activations.shape) == 2:
            input_activations = input_activations.unsqueeze(0)
            
        batch_size, sequence_length, num_in_channels = input_activations.shape
        
        input_activations = input_activations.type(torch.float32)
        
        if len(input_activations.shape) == 3:
            input_activations = input_activations.view(-1, num_in_channels)
        input_activations = input_activations.to(self.average_input_activations_sqrd_norm.device)
        
        sum_of_input_activations_sqrd_norms = (
            (self.average_input_activations_sqrd_norm * self.num_samples) 
            + (torch.norm(input_activations, p=2, dim=0).pow(2))
        )
        
        self.num_samples += batch_size
        self.average_input_activations_sqrd_norm = sum_of_input_activations_sqrd_norms / self.num_samples
        
        
class OutputProjectionLayerWrapper:
    """This class wraps a proj_out layer for specific operations."""
    def __init__(self, layer: torch.nn.Linear):
        self.layer = layer
        num_out_channels, num_in_channels = layer.weight.data.shape

        self.average_input_activations_sqrd_norm = torch.zeros((num_in_channels,), device=layer.weight.device)
        self.input_activations = None
        self.num_samples = 0
        
        
    def add_batch(self, input_activations: torch.Tensor, output_activations: torch.Tensor):
        if len(input_activations.shape) == 2:
            input_activations = input_activations.unsqueeze(0)
        
        self.input_activations = input_activations  # save it for the Value Projection Layer
        
        input_activations = input_activations.type(torch.float32)
        if len(input_activations.shape) == 3:
            input_activations = input_activations.view(-1, input_activations.shape[-1])
        input_activations = input_activations.to(self.average_input_activations_sqrd_norm.device)
        
        sum_of_input_activations_sqrd_norms = (
            (self.average_input_activations_sqrd_norm * self.num_samples) 
            + (torch.norm(input_activations, p=2, dim=0).pow(2))
        )
        
        self.num_samples += input_activations.shape[0]
        self.average_input_activations_sqrd_norm = sum_of_input_activations_sqrd_norms / self.num_samples
