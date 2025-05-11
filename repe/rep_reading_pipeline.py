from typing import List, Union, Optional, Dict, Any, Type
from transformers import Pipeline
import torch
import numpy as np
import logging
from .rep_readers import DIRECTION_FINDERS, RepReader

logger = logging.getLogger(__name__)

class RepReadingPipeline(Pipeline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("Initializing representation reading pipeline")
        # Fix Conv1d padding issue in Mamba2Block as mentioned in the memory
        self._fix_mamba_conv_padding()

    def _fix_mamba_conv_padding(self):
        """Fix the Conv1d padding issue if this is a Mamba model"""
        modified_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv1d):
                # Check if this is the Conv1d in a Mamba block with kernel_size-1 padding
                if hasattr(module, 'padding') and isinstance(module.padding, tuple) and \
                   len(module.padding) == 1 and module.padding[0] == module.kernel_size[0] - 1:
                    # Change padding from kernel_size-1 to 'same'
                    module.padding = 'same'
                    modified_count += 1
        
        if modified_count > 0:
            logger.info(f"Fixed Conv1d padding in {modified_count} Mamba layers")
    
    def _get_hidden_states(
            self, 
            outputs,
            rep_token: Union[str, int]=-1,
            hidden_layers: Union[List[int], int]=-1,
            which_hidden_states: Optional[str]=None):
        
        hidden_states_layers = {}
        
        # Check for SSM states (Mamba models)
        if hasattr(outputs, 'ssm_states') and outputs.ssm_states is not None:
            # Using SSM states directly
            for layer in hidden_layers:
                layer_idx = layer if layer >= 0 else len(outputs.ssm_states) + layer
                if 0 <= layer_idx < len(outputs.ssm_states):
                    hidden_states = outputs.ssm_states[layer_idx]
                    # For rep_token=-1, get the last token's state
                    token_idx = rep_token if rep_token >= 0 else hidden_states.size(1) + rep_token
                    hidden_states = hidden_states[:, token_idx, :].detach()
                    if hidden_states.dtype == torch.bfloat16:
                        hidden_states = hidden_states.float()
                    hidden_states_layers[layer] = hidden_states.detach()
            return hidden_states_layers
            
        # Fallback to standard hidden_states if SSM states not available
        if 'hidden_states' in outputs:
            for layer in hidden_layers:
                layer_idx = layer if layer >= 0 else len(outputs['hidden_states']) + layer
                if 0 <= layer_idx < len(outputs['hidden_states']):
                    hidden_states = outputs['hidden_states'][layer_idx]
                    token_idx = rep_token if rep_token >= 0 else hidden_states.size(1) + rep_token
                    hidden_states = hidden_states[:, token_idx, :].detach()
                    if hidden_states.dtype == torch.bfloat16:
                        hidden_states = hidden_states.float()
                    hidden_states_layers[layer] = hidden_states.detach()
        
        # If nothing worked, try to find hidden states in other common attributes
        if not hidden_states_layers:
            possible_attrs = ['last_hidden_state', 'all_hidden_states']
            for attr_name in possible_attrs:
                if hasattr(outputs, attr_name):
                    hidden_states_source = getattr(outputs, attr_name)
                    if isinstance(hidden_states_source, (list, tuple)):
                        for layer in hidden_layers:
                            layer_idx = layer if layer >= 0 else len(hidden_states_source) + layer
                            if 0 <= layer_idx < len(hidden_states_source):
                                hidden_states = hidden_states_source[layer_idx]
                                token_idx = rep_token if rep_token >= 0 else hidden_states.size(1) + rep_token
                                hidden_states = hidden_states[:, token_idx, :].detach()
                                if hidden_states.dtype == torch.bfloat16:
                                    hidden_states = hidden_states.float()
                                hidden_states_layers[layer] = hidden_states.detach()
                    break
                    
        if not hidden_states_layers:
            logger.warning("Could not find hidden states in Mamba model output. Check model output format.")

        return hidden_states_layers

    def _sanitize_parameters(self, 
                             rep_reader: RepReader=None,
                             rep_token: Union[str, int]=-1,
                             hidden_layers: Union[List[int], int]=-1,
                             component_index: int=0,
                             which_hidden_states: Optional[str]=None,
                             **tokenizer_kwargs):
        preprocess_params = tokenizer_kwargs
        forward_params =  {}
        postprocess_params = {}

        forward_params['rep_token'] = rep_token

        if not isinstance(hidden_layers, list):
            hidden_layers = [hidden_layers]


        assert rep_reader is None or len(rep_reader.directions) == len(hidden_layers), f"expect total rep_reader directions ({len(rep_reader.directions)})== total hidden_layers ({len(hidden_layers)})"                 
        forward_params['rep_reader'] = rep_reader
        forward_params['hidden_layers'] = hidden_layers
        forward_params['component_index'] = component_index
        forward_params['which_hidden_states'] = which_hidden_states
        
        return preprocess_params, forward_params, postprocess_params
 
    def preprocess(
            self, 
            inputs: Union[str, List[str], List[List[str]]],
            **tokenizer_kwargs):

        if self.image_processor:
            return self.image_processor(inputs, add_end_of_utterance_token=False, return_tensors="pt")
        return self.tokenizer(inputs, return_tensors=self.framework, **tokenizer_kwargs)

    def postprocess(self, outputs):
        return outputs

    def _forward(self, model_inputs, rep_token, hidden_layers, rep_reader=None, component_index=0, which_hidden_states=None, pad_token_id=None):
        """Forward pass for model"""
        # Run the model and get hidden states
        with torch.no_grad():
            # Ensure the proper config for model
            forward_kwargs = {
                'output_hidden_states': True,
                'return_dict': True
            }
            
            # Move inputs to the same device as the model
            if hasattr(self.model, 'device') and hasattr(model_inputs, 'to'):
                model_inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v 
                               for k, v in model_inputs.items()}
            
            # Get model outputs
            outputs = self.model(**model_inputs, **forward_kwargs)
                
        hidden_states = self._get_hidden_states(outputs, rep_token, hidden_layers, which_hidden_states)
        
        if rep_reader is None:
            return hidden_states
        
        return rep_reader.transform(hidden_states, hidden_layers, component_index)


    def _batched_string_to_hiddens(self, train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, **tokenizer_args):
        # Wrapper method to get a dictionary hidden states from a list of strings
        hidden_states_outputs = self(train_inputs, rep_token=rep_token,
            hidden_layers=hidden_layers, batch_size=batch_size, rep_reader=None, which_hidden_states=which_hidden_states, **tokenizer_args)
        hidden_states = {layer: [] for layer in hidden_layers}
        for hidden_states_batch in hidden_states_outputs:
            for layer in hidden_states_batch:
                hidden_states[layer].extend(hidden_states_batch[layer])
        return {k: np.vstack(v) for k, v in hidden_states.items()}
    
    def _validate_params(self, n_difference, direction_method):
        # validate params for get_directions
        if direction_method == 'clustermean':
            assert n_difference == 1, "n_difference must be 1 for clustermean"

    def get_directions(
            self, 
            train_inputs: Union[str, List[str], List[List[str]]], 
            rep_token: Union[str, int]=-1, 
            hidden_layers: Union[str, int]=-1,
            n_difference: int = 1,
            batch_size: int = 8, 
            train_labels: List[int] = None,
            direction_method: str = 'pca',
            direction_finder_kwargs: dict = {},
            which_hidden_states: Optional[str]=None,
            **tokenizer_args,):
        """Train a RepReader on the training data.
        Args:
            batch_size: batch size to use when getting hidden states
            direction_method: string specifying the RepReader strategy for finding directions
            direction_finder_kwargs: kwargs to pass to RepReader constructor
        """

        if not isinstance(hidden_layers, list): 
            assert isinstance(hidden_layers, int)
            hidden_layers = [hidden_layers]
        
        self._validate_params(n_difference, direction_method)

        # initialize a DirectionFinder
        direction_finder = DIRECTION_FINDERS[direction_method](**direction_finder_kwargs)

		# if relevant, get the hidden state data for training set
        hidden_states = None
        relative_hidden_states = None
        if direction_finder.needs_hiddens:
            # get raw hidden states for the train inputs
            hidden_states = self._batched_string_to_hiddens(train_inputs, rep_token, hidden_layers, batch_size, which_hidden_states, **tokenizer_args)
            
            # get differences between pairs
            relative_hidden_states = {k: np.copy(v) for k, v in hidden_states.items()}
            for layer in hidden_layers:
                for _ in range(n_difference):
                    relative_hidden_states[layer] = relative_hidden_states[layer][::2] - relative_hidden_states[layer][1::2]

		# get the directions
        direction_finder.directions = direction_finder.get_rep_directions(
            self.model, self.tokenizer, relative_hidden_states, hidden_layers,
            train_choices=train_labels)
        for layer in direction_finder.directions:
            if type(direction_finder.directions[layer]) == np.ndarray:
                direction_finder.directions[layer] = direction_finder.directions[layer].astype(np.float32)

        if train_labels is not None:
            direction_finder.direction_signs = direction_finder.get_signs(
            hidden_states, train_labels, hidden_layers)
        
        return direction_finder
