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
    # Handle empty or invalid inputs
    if direction is None or (hasattr(direction, 'size') and direction.size == 0) or \
       (hasattr(direction, 'numel') and direction.numel() == 0):
        logger.error("Received empty direction vector in project_onto_direction")
        if isinstance(H, torch.Tensor):
            return torch.zeros(H.shape[0], device=H.device)
        else:
            return np.zeros(len(H))
    
    # Ensure H and direction are proper tensors on the same device
    if not isinstance(H, torch.Tensor):
        H = torch.tensor(H, dtype=torch.float32)
        # Only move to CUDA if available
        if torch.cuda.is_available():
            H = H.cuda()
    
    if not isinstance(direction, torch.Tensor):
        try:
            direction = torch.tensor(direction, dtype=torch.float32)
            direction = direction.to(H.device)
        except Exception as e:
            logger.error(f"Failed to convert direction to tensor: {e}")
            return torch.zeros(H.shape[0], device=H.device)
    
    # Ensure the direction has proper dimensions for matmul
    if direction.dim() == 0:  # scalar
        logger.warning("Direction is a scalar, creating proper vector")
        direction = torch.ones(H.shape[1], device=H.device) * direction
    elif direction.dim() == 1:  # vector
        if direction.shape[0] != H.shape[1]:
            logger.error(f"Direction vector dimension {direction.shape[0]} doesn't match hidden state dimension {H.shape[1]}")
            return torch.zeros(H.shape[0], device=H.device)
    elif direction.dim() == 2:  # matrix - ensure it's a single vector
        if direction.shape[0] == 1:  # row vector
            direction = direction.squeeze(0)
        elif direction.shape[1] == 1:  # column vector
            direction = direction.squeeze(1)
        else:
            logger.error(f"Direction has invalid shape {direction.shape}")
            return torch.zeros(H.shape[0], device=H.device)
    
    # Calculate magnitude, with safety check
    mag = torch.norm(direction)
    if torch.isinf(mag).any() or mag < 1e-8:
        logger.warning("Direction vector has extremely small or infinite magnitude")
        mag = torch.clamp(mag, min=1e-8)  # Prevent division by zero
    
    # Debug info
    logger.info(f"H shape: {H.shape}, direction shape: {direction.shape}")
    
    try:
        # Calculate the projection safely
        projection = H.matmul(direction) / mag
        return projection
    except Exception as e:
        logger.error(f"Error in projection calculation: {e}")
        return torch.zeros(H.shape[0], device=H.device)

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
            
            # Get hidden dimension size first
            if isinstance(hidden_states[layer], torch.Tensor):
                H_train = hidden_states[layer].cpu().numpy()
            else:
                H_train = np.array(hidden_states[layer])
            
            hidden_size = H_train.shape[1]
            logger.info(f"Layer {layer} - Hidden size: {hidden_size}")
            
            # Handle case where one class has no examples
            if len(neg_class[0]) == 0 or len(pos_class[0]) == 0:
                logger.warning(f"One class has no examples in layer {layer}. Using random direction.")
                # Create a properly shaped random direction vector
                directions[layer] = np.random.randn(1, hidden_size)
                continue

            # Apply robust normalization
            H_train = H_train / (np.linalg.norm(H_train, axis=1, keepdims=True) + 1e-8)

            # Calculate mean vectors for positive and negative classes
            H_pos_mean = H_train[pos_class].mean(axis=0, keepdims=True)
            H_neg_mean = H_train[neg_class].mean(axis=0, keepdims=True)

            # The direction is from negative to positive class
            direction = H_pos_mean - H_neg_mean
            
            # Check that the direction has the right shape
            if direction.shape != (1, hidden_size):
                logger.warning(f"Direction has wrong shape {direction.shape}, reshaping to (1, {hidden_size})")
                if len(direction.shape) == 1:
                    # Create a properly shaped direction vector
                    if direction.shape[0] == 1:
                        # This is the problematic case - a scalar in a 1D array
                        new_direction = np.zeros((1, hidden_size))
                        new_direction[0, 0] = direction[0]  # Place the value in the first position
                        direction = new_direction
                    else:
                        # This is a vector, reshape to row vector
                        direction = direction.reshape(1, -1)
                        # If it doesn't match hidden_size, pad or truncate
                        if direction.shape[1] != hidden_size:
                            new_direction = np.zeros((1, hidden_size))
                            copy_size = min(direction.shape[1], hidden_size)
                            new_direction[0, :copy_size] = direction[0, :copy_size]
                            direction = new_direction
                elif direction.shape[0] != 1:
                    direction = direction.mean(axis=0, keepdims=True)
            
            # Normalize the direction vector
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                logger.warning(f"Direction vector for layer {layer} has near-zero norm. Using random direction.")
                direction = np.random.randn(1, hidden_size)
            else:
                direction = direction / norm
            
            # Verify final shape is correct
            if direction.shape != (1, hidden_size):
                logger.error(f"Final direction still has wrong shape {direction.shape}. Fixing to (1, {hidden_size})")
                direction = np.random.randn(1, hidden_size)  # Last resort fallback
                
            directions[layer] = direction
        
        return directions
        
    def get_signs(self, hidden_states, train_labels, hidden_layers):
        """Determine whether the negative or positive direction corresponds to "honesty"
        Return sign array (n_components=1) for each layer.
        """
        signs = {}
        
        for layer in hidden_layers:
            if layer not in hidden_states or layer not in self.directions:
                logger.warning(f"Layer {layer} missing in hidden_states or directions. Skipping sign determination.")
                signs[layer] = np.array([1])  # Default to positive sign
                continue
                
            # Get hidden states
            if isinstance(hidden_states[layer], torch.Tensor):
                H_train = hidden_states[layer].cpu().numpy()
            else:
                H_train = np.array(hidden_states[layer])
                
            # Get direction and ensure it has the right shape
            direction = self.directions[layer]
            
            # Print debug info about shapes
            logger.info(f"Layer {layer} - H_train shape: {H_train.shape}, direction shape: {direction.shape}, labels length: {len(train_labels)}")
            
            # Skip if shapes are incompatible and set default sign
            if len(direction.shape) <= 1 and direction.shape[0] == 1:
                logger.warning(f"Direction for layer {layer} has invalid shape {direction.shape}. Using default sign.")
                signs[layer] = np.array([1])
                continue
            
            # For 1D vectors, reshape to 2D
            if len(direction.shape) == 1:
                direction = direction.reshape(1, -1)
            
            # Check if last dimension matches
            feature_dim = H_train.shape[1]
            if direction.shape[1] != feature_dim:
                logger.warning(f"Direction feature dim {direction.shape[1]} doesn't match hidden states dim {feature_dim}. Using default sign.")
                signs[layer] = np.array([1])
                continue
            
            # Normalize for stability
            H_train = H_train / (np.linalg.norm(H_train, axis=1, keepdims=True) + 1e-8)
            
            try:
                # Project hidden states onto direction
                projections = np.dot(H_train, direction.T)
                
                # Get labels for each entry - handle size mismatch
                train_labels_np = np.array(train_labels)
                
                # Handle size mismatch between hidden states and labels
                if len(train_labels_np) != len(H_train):
                    logger.warning(f"Size mismatch: labels ({len(train_labels_np)}) vs hidden states ({len(H_train)}). Using first {min(len(train_labels_np), len(H_train))} examples.")
                    
                    # Use as many labels as possible without error
                    size = min(len(train_labels_np), len(H_train))
                    truncated_labels = train_labels_np[:size]
                    truncated_projections = projections[:size]
                    
                    # Calculate positive and negative indices on the truncated data
                    pos_indices = truncated_labels == 1
                    neg_indices = truncated_labels == 0
                else:
                    # Normal case where sizes match
                    pos_indices = train_labels_np == 1
                    neg_indices = train_labels_np == 0
                    truncated_projections = projections
                
                if np.any(pos_indices) and np.any(neg_indices):
                    # Use the truncated projections for mean calculation
                    pos_mean = np.mean(truncated_projections[pos_indices])
                    neg_mean = np.mean(truncated_projections[neg_indices])
                    # Determine sign: if positive directions correlate with positive labels,
                    # sign should be positive; otherwise negative
                    sign = 1 if pos_mean > neg_mean else -1
                    logger.info(f"Layer {layer} - pos_mean: {pos_mean:.4f}, neg_mean: {neg_mean:.4f}, sign: {sign}")
                else:
                    # Default to positive if no comparison can be made
                    logger.warning(f"No positive or negative examples for layer {layer}. Using default sign.")
                    sign = 1
            except Exception as e:
                logger.error(f"Error computing sign for layer {layer}: {str(e)}. Using default sign.")
                sign = 1
                
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