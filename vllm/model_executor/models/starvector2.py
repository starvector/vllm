from vllm.attention import AttentionMetadata
import torch.nn as nn
import torch
from typing import List
from transformers import AutoTokenizer
from vllm.model_executor.models.starcoder2 import Starcoder2ForCausalLM
from starvector.model.image_encoder.image_encoder import ImageEncoder
from vllm.model_executor.models.starvector import Adapter

from transformers import AutoConfig

from transformers import AutoImageProcessor
from transformers.processing_utils import ProcessorMixin
from transformers import BatchFeature

class StarVector2Processor(ProcessorMixin):
    attributes = ["tokenizer"]  # Only include tokenizer in attributes
    # valid_kwargs = ["size", "mean", "std"]  # Additional parameters allowed
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, 
                 tokenizer=None,  # Make tokenizer the first argument
                 **kwargs):
        model_name = tokenizer.name_or_path
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_name,
            **kwargs,
        )
        self.size = self.image_processor.size
        self.mean = self.image_processor.image_mean
        self.std = self.image_processor.image_std
        super().__init__(tokenizer=tokenizer)

    def __call__(self, images=None, text=None, **kwargs) -> BatchFeature:
        """
        Process images and/or text inputs.
        
        Args:
            images: Optional image input(s)
            text: Optional text input(s)
            **kwargs: Additional arguments for the image processor or tokenizer.
        
        Returns:
            BatchFeature containing both image and text features.
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images, **kwargs)  # Let the AutoImageProcessor handle images
        
        text_inputs = {}
        if text is not None:
            text_inputs = self.tokenizer(text, **kwargs)
        
        return BatchFeature(data={**text_inputs, **image_inputs})



class StarCoderModel(nn.Module):
    def __init__(self, vllm_config, prefix: str = "", **kwargs):
        super(StarCoderModel, self).__init__()

        self.init_tokenizer(vllm_config.model_config.hf_config.starcoder_model_name)

        self.max_length = vllm_config.model_config.hf_config.max_length
        model_config = AutoConfig.from_pretrained(vllm_config.model_config.hf_config.starcoder_model_name)

        kwargs = {}
        # kwargs["trust_remote_code"] = True

        # Configure special tokens for generation
        model_config.eos_token_id = self.tokenizer.eos_token_id
        model_config.pad_token_id = self.tokenizer.pad_token_id
        model_config.bos_token_id = self.tokenizer.bos_token_id
        model_config.vocab_size = len(self.tokenizer)

        # vllm_config.model_config.hf_config.n_inner = model_config.n_inner
        # Update vocabulary size to match the tokenizer
        vllm_config.model_config.hf_config.vocab_size = len(self.tokenizer)
        # Map hidden size from starcoder2 config to vllm config
        
        vllm_config.model_config.hf_config.hidden_size = model_config.hidden_size
        vllm_config.model_config.hf_config.layer_norm_epsilon = model_config.layer_norm_epsilon
        vllm_config.model_config.hf_config.max_position_embeddings = model_config.max_position_embeddings
        vllm_config.model_config.hf_config.activation_function = model_config.activation_function
        vllm_config.model_config.hf_config.num_key_value_heads = model_config.num_key_value_heads
        vllm_config.model_config.hf_config.rope_theta = model_config.rope_theta
        vllm_config.model_config.hf_config.use_bias = model_config.use_bias
        vllm_config.model_config.hf_config.num_key_value_heads = model_config.num_key_value_heads
        vllm_config.model_config.hf_config.intermediate_size = model_config.intermediate_size
        vllm_config.model_config.hf_config.hidden_act = model_config.hidden_act
        vllm_config.model_config.hf_config.norm_epsilon = model_config.layer_norm_epsilon

        model = Starcoder2ForCausalLM(vllm_config=vllm_config, prefix=prefix, **kwargs)
        self.transformer = model

        # Prompt the model after image
        self.prompt = "<svg"


    def init_tokenizer(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        # Incude padding and eos tokens in the vocabulary
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.add_special_tokens({"eos_token": "[EOS]"})
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})       
        
        self.svg_start_token = "<svg-start>"
        self.svg_end_token = "<svg-end>"
        self.image_start_token = "<image-start>"
        self.text_start_token = "<caption-start>"
        
        self.tokenizer.add_tokens([self.svg_start_token, self.image_start_token, self.text_start_token, self.svg_end_token])
        self.svg_start_token_id = self.tokenizer.encode(self.svg_start_token)[0]
        self.svg_end_token_id = self.tokenizer.encode(self.svg_end_token)[0]
        self.image_start_token_id = self.tokenizer.encode(self.image_start_token)[0]
        self.text_start_token_id = self.tokenizer.encode(self.text_start_token)[0]
        self.tokenizer.padding_side = "left"

class StarVector2Model(nn.Module):
    def __init__(self, vllm_config, prefix: str = "", **kwargs):
        super().__init__()
        config = vllm_config.model_config.hf_config
        config.torch_dtype = vllm_config.model_config.dtype
        self.model_precision = config.torch_dtype

        # Build Code LLM (StarCoder)
        self.svg_transformer = StarCoderModel(vllm_config, prefix=prefix, **kwargs)

        # Task-specific layers
        self.task = kwargs.get("task", "im2svg")
        self.query_length = 0
        if self.use_image_encoder():
            # Build Image Encoder
            self.image_encoder = ImageEncoder(config, **kwargs)

            # Build Adapter
            self.image_projection = self.get_adapter(config, **kwargs).to(dtype=self.model_precision)

        self.max_length = config.max_length_train - self.query_length - 3  # for image, text, and svg tokens

    def get_adapter(self, config, **kwargs):
        vision_hidden_size, self.query_length = self.get_hidden_size_and_query_length(config.image_encoder_type)
        llm_hidden_size = self.svg_transformer.transformer.config.hidden_size
        image_projection = Adapter(
            vision_hidden_size,
            llm_hidden_size,
            adapter_norm=config.adapter_norm,
            query_length=self.query_length,
            dropout_prob=kwargs.get('dropout', 0.1)
        )
        return image_projection
    
    def use_image_encoder(self):
        if (
            self.task == "im2svg"
            or self.predict_visual_tokens
            or self.task == "im-or-text2svg"
        ):
            return True
        else:
            return False

    def get_hidden_size_and_query_length(self, image_encoder_type):
        if image_encoder_type == 'clip':
            hidden_size = self.image_encoder.visual_encoder.num_features
            query_length = 257
        elif 'siglip' in image_encoder_type:
            hidden_size = (self.image_encoder.visual_encoder.head.mlp.fc2.out_features)
            if '512' in image_encoder_type:
                query_length = 1024
            elif '384' in image_encoder_type:
                query_length = 576
        return hidden_size, query_length
    
    def embed_image(self, images):
        # Process image
        embedded_image = self.image_encoder(images)
        conditioning_embeds = self.image_projection(embedded_image)
        return conditioning_embeds

    def _parse_and_validate_image_input(self, **kwargs):
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is not None and pixel_values.dim() == 5 and pixel_values.size(1) == 1:
            pixel_values = pixel_values.squeeze(1)
        return pixel_values

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs: object,
    ) -> torch.Tensor:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is not None:
            visual_embeddings = self.embed_image(image_input)
            inputs_embeds = self.svg_transformer.transformer.model.embed_tokens(input_ids)
            mask = input_ids == self.svg_transformer.image_start_token_id
            inputs_embeds[mask] = visual_embeddings.view(-1, visual_embeddings.size(-1))
        else:
            inputs_embeds = None
        # tokenizer = AutoTokenizer.from_pretrained("ServiceNow/starvector2-8b-text2svg-svg-stack-v1")
        hidden_states = self.svg_transformer.transformer(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states
