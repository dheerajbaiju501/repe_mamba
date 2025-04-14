from typing import List, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .rep_readers import DIRECTION_FINDERS, RepReader
from .utils import ensure_tensor_on_device

# Monkey patch the Linear module to ensure tensors are on the correct device
original_linear_forward = nn.Linear.forward

def device_safe_linear_forward(self, input):
    # Make sure all tensors are on the same device before operations
    if input.device != self.weight.device:
        print(f"Moving input tensor from {input.device} to {self.weight.device} in Linear.forward")
        input = input.to(self.weight.device)
    return original_linear_forward(self, input)

# Apply the monkey patch
nn.Linear.forward = device_safe_linear_forward

class Mamba2Block(nn.Module):
    def __init__(self, hidden_dim, ssm_state_dim=16, expand_factor=2, kernel_size=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ssm_state_dim = ssm_state_dim
        self.expanded_dim = hidden_dim * expand_factor
        
        # Mamba-2 specific components
        # 1. Input expansion and conv
        self.in_proj = nn.Linear(hidden_dim, self.expanded_dim * 2)  # for x and delta_x
        self.conv1d = nn.Conv1d(self.expanded_dim, self.expanded_dim, kernel_size=kernel_size, padding='same', groups=self.expanded_dim)
        
        # 2. SSM parameters with selective state spaces
        self.A = nn.Parameter(torch.randn(self.expanded_dim, ssm_state_dim))
        self.B = nn.Parameter(torch.randn(self.expanded_dim, ssm_state_dim))
        self.C = nn.Parameter(torch.randn(self.expanded_dim, ssm_state_dim))
        self.D = nn.Parameter(torch.randn(self.expanded_dim))  # Direct path D
        
        # 3. Gating mechanisms
        self.out_gate = nn.Linear(self.expanded_dim, hidden_dim)
        self.time_mix = nn.Parameter(torch.randn(self.expanded_dim))
        
        # 4. Layer norm for better stability
        self.norm = nn.LayerNorm(hidden_dim)
        
    def selective_scan(self, x, delta):
        # Mamba-2's selective scan operation
        batch, seq_len, dim = x.shape
        h = torch.zeros(batch, dim, self.ssm_state_dim, device=x.device)
        
        outputs = []
        for t in range(seq_len):
            # Time mixing
            xt = x[:, t]
            dt = delta[:, t]
            
            # Selective state update with delta-based gating
            A_hat = torch.exp(self.A * dt.unsqueeze(-1))
            h = A_hat * h + self.B * xt.unsqueeze(-1)
            
            # Output projection with selective features
            y = (self.C * h).sum(-1) + self.D * xt
            outputs.append(y)
            
        return torch.stack(outputs, dim=1)

    def forward(self, u):
        # 1. Input normalization
        u = self.norm(u)
        
        # 2. Input projection and splitting
        x_and_delta = self.in_proj(u)
        x, delta = x_and_delta.chunk(2, dim=-1)
        print("x shape:", x.shape)
        
        
        # 3. Convolutional processing
        x_conv = self.conv1d(x.transpose(-1, -2)).transpose(-1, -2)
        delta = F.silu(delta)  # Delta gating with SiLU
        
        # 4. Time mixing
        x_mix = x_conv * self.time_mix + x * (1 - self.time_mix)
        
        # 5. Selective scan operation (core Mamba-2 mechanism)
        y = self.selective_scan(x_mix, delta)
        
        # 6. Output projection with gating
        output = self.out_gate(y)
        
        return output

class Mamba2ReadingPipeline(nn.Module):
    def __init__(self, model=None, tokenizer=None, hidden_dim=768, ssm_state_dim=16, num_layers=2, **kwargs):
        """
        A Mamba-2 based pipeline that implements a similar interface to the transformers Pipeline.
        
        Args:
            model: Not used directly but kept for compatibility with transformers Pipeline interface
            tokenizer: Tokenizer to use for text processing
            hidden_dim: Hidden dimension for Mamba2 model
            ssm_state_dim: State dimension for SSM
            num_layers: Number of Mamba2 layers
        """
        super(Mamba2ReadingPipeline, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Set device first, before creating any modules or tensors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Mamba2ReadingPipeline on device: {self.device}")
        
        # Create each layer explicitly, then create the ModuleList
        layers = []
        for _ in range(num_layers):
            layers.append(Mamba2Block(hidden_dim=hidden_dim, ssm_state_dim=ssm_state_dim))
        self.layers = nn.ModuleList(layers)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_in = nn.Linear(hidden_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, hidden_dim)
        
        # Store tokenizer for preprocessing
        self.tokenizer = tokenizer
        
        # Explicitly move all modules to the device
        self.to(self.device)
        
    def to(self, device):
        """Move the model to the specified device"""
        self.device = device
        print(f"Moving Mamba2ReadingPipeline to device: {device}")
        return super().to(device)
    
    def _get_hidden_states(
            self, 
            inputs,
            rep_token: Union[str, int]=-1,
            hidden_layers: Union[List[int], int]=-1,
            which_hidden_states: Optional[str]=None):
        
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            
        # Project input - ensure on correct device
        inputs = inputs.to(self.device)
        x = self.proj_in(inputs)
        x = self.norm(x)
        
        # Apply Mamba-2 layers and collect hidden states
        all_hidden_states = [x]  # Input embedding as first hidden state
        for i, layer in enumerate(self.layers):
            x = layer(x)
            all_hidden_states.append(x)
        
        # Get representation from specified position
        if rep_token == -1:
            hidden_states = [hs.mean(dim=1) for hs in all_hidden_states]  # Average pooling
        else:
            hidden_states = [hs[:, rep_token, :] for hs in all_hidden_states]
            
        # Create dictionary with requested layers
        hidden_states_layers = {}
        
        if not isinstance(hidden_layers, list):
            hidden_layers = [hidden_layers]
            
        for layer_idx in hidden_layers:
            # Handle negative indexing
            if layer_idx < 0:
                actual_idx = len(all_hidden_states) + layer_idx
            else:
                actual_idx = layer_idx
                
            # Ensure the index is valid
            if 0 <= actual_idx < len(all_hidden_states):
                hidden_state = hidden_states[actual_idx]
                if hidden_state.dtype == torch.bfloat16:
                    hidden_state = hidden_state.float()
                hidden_states_layers[layer_idx] = hidden_state.detach().to(self.device)

        return hidden_states_layers
        
    def force_to_device(self, obj, device=None):
        """Recursively move all tensors in an object to the specified device."""
        if device is None:
            device = self.device
            
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self.force_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.force_to_device(v, device) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.force_to_device(v, device) for v in obj)
        else:
            return obj
            
    def _forward(self, batch_dict, rep_token, hidden_layers, rep_reader=None, component_index=0, which_hidden_states=None):
        model_inputs = batch_dict["inputs"]
        
        # Force inputs to the correct device
        model_inputs = self.force_to_device(model_inputs)
        
        hidden_states = self._get_hidden_states(
            model_inputs, 
            rep_token=rep_token, 
            hidden_layers=hidden_layers, 
            which_hidden_states=which_hidden_states
        )
        
        if rep_reader is None:
            return hidden_states
        
        # Move rep_reader to the correct device if needed
        if hasattr(rep_reader, 'directions'):
            rep_reader.directions = self.force_to_device(rep_reader.directions)
            
        return rep_reader.transform(hidden_states, hidden_layers, component_index)

    def _sanitize_parameters(self, 
                           rep_reader: RepReader=None,
                           rep_token: Union[str, int]=-1,
                           hidden_layers: Union[List[int], int]=-1,
                           component_index: int=0,
                           which_hidden_states: Optional[str]=None,
                           batch_size: int=8,
                           **kwargs):
        preprocess_params = {**kwargs, "batch_size": batch_size}
        forward_params = {
            'rep_token': rep_token,
            'rep_reader': rep_reader,
            'hidden_layers': [hidden_layers] if not isinstance(hidden_layers, list) else hidden_layers,
            'component_index': component_index,
            'which_hidden_states': which_hidden_states
        }
        postprocess_params = {}
        
        if rep_reader is not None:
            assert len(rep_reader.directions) == len(forward_params['hidden_layers']), \
                f"Expect total rep_reader directions ({len(rep_reader.directions)}) == total hidden_layers ({len(forward_params['hidden_layers'])})"
        
        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, inputs: Union[str, List[str], List[List[str]]], batch_size=8, **kwargs):
        # Convert inputs to batches
        if isinstance(inputs, str):
            inputs = [inputs]
            
        batches = [inputs[i:i+batch_size] for i in range(0, len(inputs), batch_size)]
        
        processed_batches = []
        for batch in batches:
            if self.tokenizer:
                # Use tokenizer if available
                encoded = self.tokenizer(batch, return_tensors="pt", padding=True, **kwargs)
                # Move inputs to the correct device immediately after tokenization
                encoded = {k: v.to(self.device) for k, v in encoded.items() if isinstance(v, torch.Tensor)}
                
                # Create embeddings - ensure they're on the correct device
                # Note: In a real implementation this would use a proper embedding layer
                embeddings = torch.randn(len(batch), 20, self.hidden_dim, device=self.device)
                processed_batches.append({"inputs": embeddings, "batch": batch})
            else:
                # Simple fallback encoding if no tokenizer
                processed_batches.append({
                    "inputs": torch.stack([self._encode_text(text) for text in batch]).to(self.device),
                    "batch": batch
                })
                
        return processed_batches

    def _encode_text(self, text: str) -> torch.Tensor:
        # This is a simplified encoding function - in practice you'd want something more sophisticated
        # Return random vectors on the correct device
        return torch.randn(20, self.hidden_dim, device=self.device)

    def postprocess(self, outputs):
        return outputs

    def __call__(self, inputs, **kwargs):
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)
        
        # Ensure the model is on the correct device
        self.to(self.device)
        
        batch_size = preprocess_params.pop("batch_size", 8)
        processed_batches = self.preprocess(inputs, batch_size=batch_size, **preprocess_params)
        
        outputs = []
        for batch_dict in processed_batches:
            # Ensure batch inputs are on the correct device
            if "inputs" in batch_dict and isinstance(batch_dict["inputs"], torch.Tensor):
                batch_dict["inputs"] = batch_dict["inputs"].to(self.device)
                
            model_outputs = self._forward(batch_dict, **forward_params)
            
            # Ensure model outputs are properly processed
            if isinstance(model_outputs, dict):
                for key in model_outputs:
                    if isinstance(model_outputs[key], torch.Tensor):
                        # Keep on same device as model
                        model_outputs[key] = model_outputs[key].to(self.device)
                        
            outputs.append(model_outputs)
            
        if len(outputs) == 1:
            return outputs[0]
        return outputs
    
    def _batched_string_to_hiddens(self, train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, **tokenizer_args):
        """Wrapper method to get a dictionary of hidden states from a list of strings."""
        hidden_states_outputs = self(train_inputs, rep_token=rep_token,
            hidden_layers=hidden_layers, batch_size=batch_size, rep_reader=None, which_hidden_states=which_hidden_states, **tokenizer_args)
        
        # If we have multiple batches, combine them
        if isinstance(hidden_states_outputs, list):
            hidden_states = {layer: [] for layer in hidden_layers}
            for hidden_states_batch in hidden_states_outputs:
                for layer in hidden_states_batch:
                    hidden_states[layer].extend(hidden_states_batch[layer])
            # Keep on GPU - don't convert to numpy!
            return {k: torch.stack(v).to(self.device) if isinstance(v[0], torch.Tensor) else torch.tensor(np.vstack(v), device=self.device) for k, v in hidden_states.items()}
        else:
            # If only one batch, keep it on GPU
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else torch.tensor(v, device=self.device) for k, v in hidden_states_outputs.items()}
    
    def _validate_params(self, n_difference, direction_method):
        """Validate parameters for get_directions."""
        assert n_difference >= 1, "Must have at least one difference vector"
        assert direction_method in DIRECTION_FINDERS, f"Unknown direction_method, must be one of {list(DIRECTION_FINDERS.keys())}"
    
    def get_directions(self, train_inputs, rep_token=-1, hidden_layers=-1, n_difference=1, train_labels=None, direction_method='pca', 
                      batch_size=8, **tokenizer_args):
        """
        Get directions in the representation space that differentiate between inputs.
        
        Args:
            train_inputs: List of strings to compute hidden states on
            rep_token: Which token to use as representation
            hidden_layers: Hidden layers to get representations from
            n_difference: Number of difference vectors to compute
            train_labels: Labels for supervised methods
            direction_method: Method to find directions ('pca', 'supervised', etc.)
            batch_size: Batch size for processing
            **tokenizer_args: Additional arguments for tokenizer
            
        Returns:
            RepReader object with the computed directions
        """
        try:
            # Ensure model is on the correct device
            self.to(self.device)
            
            self._validate_params(n_difference, direction_method)
            
            # Convert hidden_layers to list if it's not already
            if not isinstance(hidden_layers, list):
                hidden_layers = [hidden_layers]
                
            # Get hidden states for all inputs
            hidden_states = self._batched_string_to_hiddens(
                train_inputs, rep_token, hidden_layers, batch_size, None, **tokenizer_args
            )
            
            # Make sure all hidden_states are on the correct device
            hidden_states = self.force_to_device(hidden_states)
            
            # Get the direction finder based on the method
            direction_finder = DIRECTION_FINDERS[direction_method](n_components=n_difference)
            
            # Compute the directions
            direction_finder.directions = direction_finder.get_rep_directions(
                self, self.tokenizer, hidden_states, hidden_layers, train_choices=train_labels
            )
            
            # Make sure all directions are PyTorch tensors on the correct device
            for layer in direction_finder.directions:
                if isinstance(direction_finder.directions[layer], np.ndarray):
                    direction_finder.directions[layer] = torch.tensor(direction_finder.directions[layer], device=self.device)
            
            # Set device attribute on direction_finder
            direction_finder.device = self.device
            
            # Use force_to_device to make absolutely sure all tensors are on the correct device
            direction_finder.directions = self.force_to_device(direction_finder.directions)
            
            # Store model device in the direction_finder
            if hasattr(direction_finder, 'H_train_means'):
                direction_finder.H_train_means = self.force_to_device(direction_finder.H_train_means)
                
            return direction_finder
            
        except Exception as e:
            print(f"Error in get_directions: {e}")
            # Print device information for debugging
            print(f"Device is: {self.device}")
            if torch.cuda.is_available():
                print(f"CUDA is available, current device: {torch.cuda.current_device()}")
            import traceback
            traceback.print_exc()
            raise

class SimpleMamba2ReadingPipeline:
    """Legacy class, not used."""
    pass
