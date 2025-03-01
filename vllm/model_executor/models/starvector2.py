from transformers import PretrainedConfig, PreTrainedModel
from argparse import Namespace
from typing import Iterable, List, Optional, Tuple
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.attention import Attention, AttentionMetadata

import torch.nn as nn
import torch
from torch import nn

from vllm.attention import Attention, AttentionMetadata
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearMethodBase,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput


from transformers import AutoTokenizer
from vllm.model_executor.models.starcoder2 import Starcoder2Config, Starcoder2Model, Starcoder2ForCausalLM
from starvector.model.image_encoder.image_encoder import ImageEncoder
from vllm.model_executor.models.starvector import Adapter

from transformers import AutoConfig


class StarCoderModel(nn.Module):
    def __init__(self, config):
        super(StarCoderModel, self).__init__()

        self.init_tokenizer(config.starcoder_model_name)

        self.max_length = config.max_length
        model_config = AutoConfig.from_pretrained(config.starcoder_model_name)

        # model_config = AutoConfig.from_pretrained(config.starcoder_model_name, trust_remote_code=True)
        kwargs = {}
        kwargs["trust_remote_code"] = True

        # Configure special tokens for generation
        model_config.eos_token_id = self.tokenizer.eos_token_id
        model_config.pad_token_id = self.tokenizer.pad_token_id
        model_config.bos_token_id = self.tokenizer.bos_token_id
        model_config.vocab_size = len(self.tokenizer)

        # model = GPTBigCodeForCausalLM(config=model_config)
        model = Starcoder2ForCausalLM(config=model_config)
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
    def __init__(self, config, kwargs):
        super().__init__()

        # Build Code LLM (StarCoder)
        self.svg_transformer = StarCoderModel(config)

        # Task-specific layers
        self.task = kwargs.get("task", "im2svg")
        self.model_precision = kwargs.get("model_precision",config.torch_dtype)
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
