from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from itertools import islice
import torch
import logging

logger = logging.getLogger(__name__)

def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    # Calculate the magnitude of the direction vector for model hidden states
    # Ensure H and direction are on the same device (CPU or GPU)
    if not isinstance(H, torch.Tensor):
        H = torch.tensor(H, dtype=torch.float32)
        # Only move to CUDA if available
        if torch.cuda.is_available():
            H = H.cuda()
    
    if not isinstance(direction, torch.Tensor):
        direction = torch.tensor(direction, dtype=torch.float32)
        direction = direction.to(H.device)
    
    mag = torch.norm(direction)
    if torch.isinf(mag).any() or mag < 1e-8:
        logger.warning("Direction vector has extremely small or infinite magnitude")
        mag = torch.clamp(mag, min=1e-8)  # Prevent division by zero
    
    # Calculate the projection
    projection = H.matmul(direction) / mag
    return projection

def recenter(x, mean=None):
    # Convert to tensor if not already
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
        # Only move to CUDA if available
        if torch.cuda.is_available():
            x = x.cuda()
            
    if mean is None:
        mean = torch.mean(x, axis=0, keepdims=True)
        if torch.cuda.is_available():
            mean = mean.cuda()
    else:
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32)
            if torch.cuda.is_available():
                mean = mean.cuda()
    
    return x - mean

class RepReader(ABC):
    """Class to identify and store concept directions for language models.
    
    Subclasses implement the abstract methods to identify concept directions 
    for each hidden layer via strategies including PCA, embedding vectors 
    (aka the logits method), and cluster means.

    RepReader instances are used by RepReaderPipeline to get concept scores.

    Directions can be used for downstream interventions."""

    @abstractmethod
    def __init__(self) -> None:
        self.direction_method = None
        self.directions = None # directions accessible via directions[layer][component_index]
        self.direction_signs = None # direction of high concept scores (mapping min/max to high/low)

    @abstractmethod
    def get_rep_directions(self, model, tokenizer, hidden_states, hidden_layers, **kwargs):
        """Get concept directions for each hidden layer of the model
        
        Args:
            model: Language model to get directions for
            tokenizer: Tokenizer to use
            hidden_states: Hidden states of the model on the training data (per layer)
            hidden_layers: Layers to consider

        Returns:
            directions: A dict mapping layers to direction arrays (n_components, hidden_size)
        """
        pass 

    def get_signs(self, hidden_states, train_choices, hidden_layers):
        """Given labels for the training data hidden_states, determine whether the
        negative or positive direction corresponds to low/high concept 
        (and return corresponding signs -1 or 1 for each layer and component index)
        
        NOTE: This method assumes that there are 2 entries in hidden_states per label, 
        aka len(hidden_states[layer]) == 2 * len(train_choices). For example, if 
        n_difference=1, then hidden_states here should be the raw hidden states
        rather than the relative (i.e. the differences between pairs of examples).

        Args:
            hidden_states: Hidden states of the model on the training data (per layer)
            train_choices: Labels for the training data
            hidden_layers: Layers to consider

        Returns:
            signs: A dict mapping layers to sign arrays (n_components,)
        """        
        signs = {}

        if self.needs_hiddens and hidden_states is not None and len(hidden_states) > 0:
            for layer in hidden_layers:    
                assert hidden_states[layer].shape[0] == 2 * len(train_choices), f"Shape mismatch between hidden states ({hidden_states[layer].shape[0]}) and labels ({len(train_choices)})"
                
                signs[layer] = []
                for component_index in range(self.n_components):
                    transformed_hidden_states = project_onto_direction(hidden_states[layer], self.directions[layer][component_index])
                    projected_scores = [transformed_hidden_states[i:i+2] for i in range(0, len(transformed_hidden_states), 2)]

                    outputs_min = [1 if min(o) == o[label] else 0 for o, label in zip(projected_scores, train_choices)]
                    outputs_max = [1 if max(o) == o[label] else 0 for o, label in zip(projected_scores, train_choices)]
                    
                    signs[layer].append(-1 if np.mean(outputs_min) > np.mean(outputs_max) else 1)
        else:
            for layer in hidden_layers:    
                signs[layer] = [1 for _ in range(self.n_components)]

        return signs


    def transform(self, hidden_states, hidden_layers, component_index):
        """Project the hidden states onto the concept directions in self.directions
        
        Args:
            hidden_states: dictionary with entries of dimension (n_examples, hidden_size)
            hidden_layers: list of layers to consider
            component_index: index of the component to use from self.directions
        
        Returns:
            transformed_hidden_states: dictionary with entries of dimension (n_examples,)
        """
        transformed_hidden_states = {}
        for layer in hidden_layers:
            if layer in hidden_states and layer in self.directions:
                direction = self.directions[layer][component_index]
                
                # Apply additional normalization for robustness, especially for Mamba
                if isinstance(hidden_states[layer], torch.Tensor):
                    H = hidden_states[layer]
                    # Normalize H for more stable projections
                    H_norm = torch.norm(H, dim=1, keepdim=True)
                    H = H / (H_norm + 1e-8)
                else:
                    H = hidden_states[layer]
                    # Normalize H for more stable projections
                    H_norm = np.linalg.norm(H, axis=1, keepdims=True)
                    H = H / (H_norm + 1e-8)
                
                # Project onto direction
                projected = project_onto_direction(H=H, direction=direction)
                
                if self.direction_signs is not None and layer in self.direction_signs:
                    sign = self.direction_signs[layer][component_index]
                    transformed_hidden_states[layer] = projected * sign
                else:
                    transformed_hidden_states[layer] = projected
        
        return transformed_hidden_states

class PCARepReader(RepReader):
    """Extract directions via PCA for language models"""
    needs_hiddens = True

    def __init__(self, n_components=1):
        super().__init__()
        self.n_components = n_components
        self.H_train_means = {}

    def get_rep_directions(self, model, tokenizer, hidden_states, hidden_layers, **kwargs):
        # Get PCA components for each layer - adapted for Mamba models

        directions = {}
        for layer in hidden_layers:
            if layer not in hidden_states:
                logger.warning(f"Layer {layer} not found in hidden_states. Skipping.")
                continue
                
            H_train = hidden_states[layer]
            if len(H_train) == 0:
                logger.warning(f"No hidden states found for layer {layer}. Skipping.")
                continue
                
            # Convert to numpy if tensor
            if isinstance(H_train, torch.Tensor):
                H_train = H_train.cpu().numpy()

            # get and save the mean
            self.H_train_means[layer] = np.mean(H_train, axis=0)

            # center the data
            H_train = H_train - self.H_train_means[layer]
            
            # Determine number of components based on data dimensions
            effective_n_components = min(self.n_components, H_train.shape[0], H_train.shape[1])
            if effective_n_components < self.n_components:
                logger.warning(f"Reducing PCA components from {self.n_components} to {effective_n_components} due to data dimensions")

            # calculate PCA with error handling
            try:
                pca_model = PCA(n_components=effective_n_components, whiten=False).fit(H_train)
                directions[layer] = pca_model.components_  # shape (n_components, n_features)
                self.n_components = pca_model.n_components_
            except Exception as e:
                logger.error(f"PCA failed for layer {layer}: {str(e)}")
                # Fallback to random direction if PCA fails
                hidden_size = H_train.shape[1]
                directions[layer] = np.random.randn(effective_n_components, hidden_size)
                logger.warning(f"Using random directions for layer {layer} due to PCA failure")
        
        return directions

    def get_signs(self, hidden_states, train_labels, hidden_layers):

        signs = {}

        for layer in hidden_layers:
            assert hidden_states[layer].shape[0] == len(np.concatenate(train_labels)), f"Shape mismatch between hidden states ({hidden_states[layer].shape[0]}) and labels ({len(np.concatenate(train_labels))})"
            layer_hidden_states = hidden_states[layer]

            # NOTE: since scoring is ultimately comparative, the effect of this is moot
            layer_hidden_states = recenter(layer_hidden_states, mean=self.H_train_means[layer])

            # get the signs for each component
            layer_signs = np.zeros(self.n_components)
            for component_index in range(self.n_components):

                transformed_hidden_states = project_onto_direction(layer_hidden_states, self.directions[layer][component_index]).cpu()
                
                pca_outputs_comp = [list(islice(transformed_hidden_states, sum(len(c) for c in train_labels[:i]), sum(len(c) for c in train_labels[:i+1]))) for i in range(len(train_labels))]

                # We do elements instead of argmin/max because sometimes we pad random choices in training
                pca_outputs_min = np.mean([o[train_labels[i].index(1)] == min(o) for i, o in enumerate(pca_outputs_comp)])
                pca_outputs_max = np.mean([o[train_labels[i].index(1)] == max(o) for i, o in enumerate(pca_outputs_comp)])

       
                layer_signs[component_index] = np.sign(np.mean(pca_outputs_max) - np.mean(pca_outputs_min))
                if layer_signs[component_index] == 0:
                    layer_signs[component_index] = 1 # default to positive in case of tie

            signs[layer] = layer_signs

        return signs
    

        
class ClusterMeanRepReader(RepReader):
    """Get the direction that is the difference between the mean of the positive and negative clusters
    in model hidden states."""

    n_components = 1
    needs_hiddens = True

    def __init__(self):
        super().__init__()

    def get_rep_directions(self, model, tokenizer, hidden_states, hidden_layers, **kwargs):
        # train labels is necessary to differentiate between different classes
        train_choices = kwargs.get('train_choices')
        if train_choices is None:
            raise ValueError("ClusterMeanRepReader requires train_choices to differentiate two clusters")
            
        directions = {}
        for layer in hidden_layers:
            if layer not in hidden_states:
                logger.warning(f"Layer {layer} not found in hidden_states. Skipping.")
                continue
                
            if len(hidden_states[layer]) == 0:
                logger.warning(f"No hidden states found for layer {layer}. Skipping.")
                continue
                
            if len(train_choices) != len(hidden_states[layer]):
                logger.warning(f"Shape mismatch between hidden states ({len(hidden_states[layer])}) and labels ({len(train_choices)}). Skipping layer {layer}.")
                continue

            # Convert train_choices to numpy array if it's not already
            train_choices_np = np.array(train_choices)
            neg_class = np.where(train_choices_np == 0)
            pos_class = np.where(train_choices_np == 1)
            
            # Handle case where one class has no examples
            if len(neg_class[0]) == 0 or len(pos_class[0]) == 0:
                logger.warning(f"One class has no examples in layer {layer}. Using random direction.")
                # Get hidden dimension size
                if isinstance(hidden_states[layer], torch.Tensor):
                    H_train = hidden_states[layer].cpu().numpy()
                else:
                    H_train = np.array(hidden_states[layer])
                hidden_size = H_train.shape[1]
                directions[layer] = np.random.randn(1, hidden_size)
                continue

            # Convert to numpy if tensor
            if isinstance(hidden_states[layer], torch.Tensor):
                H_train = hidden_states[layer].cpu().numpy()
            else:
                H_train = np.array(hidden_states[layer])

            # Apply robust normalization
            H_train = H_train / (np.linalg.norm(H_train, axis=1, keepdims=True) + 1e-8)

            # Calculate mean vectors for positive and negative classes
            H_pos_mean = H_train[pos_class].mean(axis=0, keepdims=True)
            H_neg_mean = H_train[neg_class].mean(axis=0, keepdims=True)

            # The direction is from negative to positive class
            direction = H_pos_mean - H_neg_mean
            
            # Normalize the direction vector
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            
            directions[layer] = direction
        
        return directions
        
    def get_signs(self, hidden_states, train_labels, hidden_layers):
        """Determine whether the negative or positive direction corresponds to "honesty"
        Return sign array (n_components=1) for each layer.
        """
        signs = {}
        
        for layer in hidden_layers:
            if layer not in hidden_states:
                continue
                
            if isinstance(hidden_states[layer], torch.Tensor):
                H_train = hidden_states[layer].cpu().numpy()
            else:
                H_train = np.array(hidden_states[layer])
                
            # Normalize for stability
            H_train = H_train / (np.linalg.norm(H_train, axis=1, keepdims=True) + 1e-8)
            
            # Project onto direction
            direction = self.directions[layer]
            projections = H_train @ direction.T
            
            # Get labels for each entry
            train_labels_np = np.array(train_labels)
            
            # Calculate mean projections for positive and negative classes
            pos_mean = projections[train_labels_np == 1].mean()
            neg_mean = projections[train_labels_np == 0].mean()
            
            # Determine sign: if positive directions correlate with positive labels,
            # sign should be positive; otherwise negative
            sign = 1 if pos_mean > neg_mean else -1
            signs[layer] = np.array([sign])
            
        return signs


class RandomRepReader(RepReader):
    """Get random directions for each hidden layer. Do not use hidden 
    states or train labels of any kind."""

    def __init__(self, needs_hiddens=True):
        super().__init__()

        self.n_components = 1
        self.needs_hiddens = needs_hiddens

    def get_rep_directions(self, model, tokenizer, hidden_states, hidden_layers, **kwargs):
        directions = {}
        for layer in hidden_layers:
            # Get the hidden dimension size from model config
            if hasattr(model, 'config') and hasattr(model.config, 'd_model'):
                # Some models use d_model
                hidden_size = model.config.d_model
            elif hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
                # Some models use hidden_size
                hidden_size = model.config.hidden_size
            elif hidden_states is not None and layer in hidden_states and hidden_states[layer].shape[1] > 0:
                # Infer from the hidden states themselves
                hidden_size = hidden_states[layer].shape[1]
            else:
                # Default fallback
                logger.warning(f"Could not determine hidden size for layer {layer}. Using default 1024.")
                hidden_size = 1024
                
            directions[layer] = np.expand_dims(np.random.randn(hidden_size), 0)

        return directions


class MambaRepReader(ClusterMeanRepReader):
    """
    Specialized RepReader for Mamba models that applies additional processing
    to account for state space model characteristics.
    """
    
    def __init__(self, n_ensembles=5):
        super().__init__()
        self.n_ensembles = n_ensembles
    
    def get_rep_directions(self, model, tokenizer, hidden_states, hidden_layers, **kwargs):
        # Get base directions using ClusterMean approach
        base_directions = super().get_rep_directions(model, tokenizer, hidden_states, hidden_layers, **kwargs)
        
        # Apply additional ensemble processing for Mamba
        ensemble_directions = {}
        for layer in hidden_layers:
            if layer not in base_directions:
                continue
                
            # Create ensemble of directions with small random perturbations
            direction_ensembles = []
            base_dir = base_directions[layer]
            
            # Original direction
            direction_ensembles.append(base_dir)
            
            # Add perturbed versions
            for _ in range(self.n_ensembles - 1):
                # Add small random noise (1% of magnitude)
                noise_scale = 0.01 * np.linalg.norm(base_dir)
                noise = np.random.randn(*base_dir.shape) * noise_scale
                perturbed_dir = base_dir + noise
                # Renormalize
                perturbed_dir = perturbed_dir / np.linalg.norm(perturbed_dir)
                direction_ensembles.append(perturbed_dir)
            
            # Average the ensemble
            ensemble_directions[layer] = np.mean(direction_ensembles, axis=0)
            
            # Renormalize
            norm = np.linalg.norm(ensemble_directions[layer])
            if norm > 0:
                ensemble_directions[layer] = ensemble_directions[layer] / norm
            
        return ensemble_directions

DIRECTION_FINDERS = {
    'pca': PCARepReader,
    'cluster_mean': ClusterMeanRepReader,
    'random': RandomRepReader,
    'mamba': MambaRepReader,
}