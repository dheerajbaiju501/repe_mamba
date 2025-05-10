# Import the specific Mamba model class - you'll need to update this import path
# based on your actual Mamba2 installation
from state_spaces import Mamba2ForCausalLM
from transformers.pipelines import PIPELINE_REGISTRY
from .rep_reading_pipeline import RepReadingPipeline
from .rep_control_pipeline import RepControlPipeline

def repe_pipeline_registry():
    # Register the pipeline for Mamba2 models only
    PIPELINE_REGISTRY.register_pipeline(
        "rep-reading",
        pipeline_class=RepReadingPipeline,
        pt_model=Mamba2ForCausalLM,
    )

    PIPELINE_REGISTRY.register_pipeline(
        "rep-control",
        pipeline_class=RepControlPipeline,
        pt_model=Mamba2ForCausalLM,
    )


