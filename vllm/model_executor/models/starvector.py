from transformers import PretrainedConfig, PreTrainedModel
from argparse import Namespace
import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, List, Optional, Tuple
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.attention import Attention, AttentionMetadata

import torch.nn as nn
import torch.nn.init as init
import torch


import torch
from torch import nn
from transformers import GPTBigCodeConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
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
from vllm.model_executor.models.gpt_bigcode import GPTBigCodeConfig, GPTBigCodeModel

from starvector.model.image_encoder.image_encoder import ImageEncoder


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Adapter(nn.Module):
    def __init__(self, input_size, output_size,  adapter_norm="layer_norm", init_type="glorot",  query_length=32, dropout_prob=0.1):
        super().__init__()
        self.query_length = query_length
        self.dropout_prob = dropout_prob
        self.adapter_norm = adapter_norm

        self.dropout = nn.Dropout(p=self.dropout_prob)
        
        self.c_fc = nn.Linear(input_size, input_size*2)
        self.act = Swish()
        self.c_proj = nn.Linear(input_size*2, output_size)
        
        if adapter_norm == "layer_norm":
            self.norm = nn.LayerNorm([self.query_length, output_size])
        elif adapter_norm == "batch_norm":
            self.norm = nn.BatchNorm1d(self.query_length)

        self.init_type = init_type.lower()
        self._initialize_weights()

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type == "glorot":
                    init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif self.init_type == "normal":
                    init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                else:
                    raise ValueError("Invalid initialization type specified.")



class GPTBigCodeForCausalLM(nn.Module):

    def __init__(
        self,
        config: GPTBigCodeConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.linear_method = linear_method
        self.transformer = GPTBigCodeModel(config, linear_method)
        self.lm_head_weight = self.transformer.wte.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.transformer(
            input_ids, positions, kv_caches, attn_metadata, inputs_embeds
        )
        return hidden_states

from transformers import AutoConfig

class StarCoderModel(nn.Module):
    def __init__(self, config):
        super(StarCoderModel, self).__init__()

        self.init_tokenizer(config.starcoder_model_name)

        self.max_length = config.max_length
        model_config = AutoConfig.from_pretrained(config.starcoder_model_name)

        kwargs = {}
        kwargs["trust_remote_code"] = True

        # Configure special tokens for generation
        model_config.eos_token_id = self.tokenizer.eos_token_id
        model_config.pad_token_id = self.tokenizer.pad_token_id
        model_config.bos_token_id = self.tokenizer.bos_token_id
        model_config.vocab_size = len(self.tokenizer)

        model = GPTBigCodeForCausalLM(config=model_config)
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
        self.image_start_token = "<image-start>"
        self.text_start_token = "<caption-start>"

        self.tokenizer.add_tokens(
            [self.svg_start_token, self.image_start_token, self.text_start_token]
        )
        self.svg_start_token_id = self.tokenizer.encode(self.svg_start_token)[0]
        self.image_start_token_id = self.tokenizer.encode(self.image_start_token)[0]
        self.text_start_token_id = self.tokenizer.encode(self.text_start_token)[0]


class StarVectorModel(nn.Module):
    def __init__(self, config, kwargs):
        super().__init__()

        # Build Code LLM (StarCoder)
        self.svg_transformer = StarCoderModel(config)

        # Task-specific layers
        self.task = kwargs.get("task", "im2svg")
        self.model_precision = kwargs.get("model_precision", torch.float16)
        self.query_length = 0
        if self.use_image_encoder():
            # Build Image Encoder
            self.image_encoder = ImageEncoder(config, **kwargs)

            # Build Adapter
            vision_hidden_size, self.query_length = (
                self.get_hidden_size_and_query_length(config.image_encoder_type)
            )
            llm_hidden_size = self.svg_transformer.transformer.config.hidden_size
            self.image_projection = Adapter(
                input_size=vision_hidden_size,
                output_size=llm_hidden_size,
                init_type=config.init_type,
                adapter_norm=config.adapter_norm,
                query_length=self.query_length,
                dropout_prob=config.dropout,
            )
            
            self.image_projection = self.image_projection.to(dtype=self.model_precision)

        self.max_length = (
            config.max_length_train - self.query_length - 3
        )  # for image, text, and svg tokens

    def use_image_encoder(self):
        if (
            self.task == "im2svg"
        ):
            return True
        else:
            return False

    def get_hidden_size_and_query_length(self, image_encoder_type):
        if image_encoder_type == "clip":
            hidden_size = self.image_encoder.visual_encoder.num_features
            query_length = 257
        return hidden_size, query_length

    def embed_text_to_svg(self, batch, device):
        captions = batch["caption"]
        svgs = batch["svg"]
        samples = [
            captions[i]
            + self.svg_transformer.svg_start_token
            + svgs[i]
            + self.svg_transformer.tokenizer.eos_token
            for i in range(len(captions))
        ]

        tokens = self.svg_transformer.tokenizer(
            samples,
            truncation=True,
            add_special_tokens=True,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        mask_svg_padding = (
            tokens.input_ids == self.svg_transformer.tokenizer.pad_token_id
        )

        if not self.predict_prompt:
            # Create a mask for true when padding tokens
            svg_token = self.svg_transformer.tokenizer(
                self.svg_transformer.svg_start_token
            )
            svg_token_indices = (tokens.input_ids == svg_token.input_ids[0]).nonzero(
                as_tuple=True
            )[1]

            # Create a mask for true when prompt
            N, S = tokens.input_ids.size()
            indices_tensor_expanded = svg_token_indices.unsqueeze(1).expand(N, S)
            indices_range = torch.arange(S, device=tokens.input_ids.device)
            mask_prompt = indices_range <= indices_tensor_expanded

            # Combine masks for target (only svg is considered in loss)
            target_mask = mask_prompt | mask_svg_padding
        else:
            target_mask = mask_svg_padding

        targets = tokens.input_ids.masked_fill(target_mask, -100)

        # Compute embeddings
        inputs_embeds = self.svg_transformer.transformer.transformer.wte(
            tokens.input_ids
        )
        attention_mask = tokens.attention_mask

        return inputs_embeds, attention_mask, targets

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
            inputs_embeds = self.svg_transformer.transformer.transformer.wte(input_ids)
            mask = input_ids == self.svg_transformer.image_start_token_id
            inputs_embeds[mask] = visual_embeddings.view(-1, visual_embeddings.size(-1))
        else:
            inputs_embeds = None

        hidden_states = self.svg_transformer.transformer(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

class StarVectorConfig(PretrainedConfig):
    model_type = "starvector"

    def __init__(
        self,
        starcoder_model_name: str = "bigcode/starcoderbase-1b",
        image_encoder_type: str = "clip",
        adapter_norm: str = "layer_norm",
        image_size: int = 224,
        max_length: int = 8192,
        max_length_train: int = 8192,
        use_flash_attn: bool = True,
        use_cache: bool = True,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 24,
        vocab_size: int = 32000,
        hidden_size: int = 1024,
        num_kv_heads: int = 4,
        **kwargs,
    ):
        self.starcoder_model_name = starcoder_model_name
        self.image_encoder_type = image_encoder_type
        self.adapter_norm = adapter_norm
        self.image_size = image_size
        self.max_length = max_length
        self.max_length_train = max_length_train
        self.use_flash_attn = use_flash_attn
        self.use_cache = use_cache
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_kv_heads = num_kv_heads

        super().__init__(**kwargs)


class StarVectorForCausalLM(PreTrainedModel):
    config_class = StarVectorConfig
    _no_split_modules = []

    def __init__(self, config: StarVectorConfig, **kwargs):
        super().__init__(config)
        self.model_name = config._name_or_path
        if 'starvector-1b' in self.model_name:
            self.model = StarVectorModel(config=config, kwargs=kwargs)
        else:   
            from vllm.model_executor.models.starvector2 import StarVector2Model
            self.model = StarVector2Model(config=config, kwargs=kwargs)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs: object,
    ) -> SamplerOutput:

        return self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            **kwargs,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        backbone_transformer = self.model.svg_transformer.transformer
        logits = backbone_transformer.logits_processor(
            backbone_transformer.lm_head_weight, hidden_states, sampling_metadata
        )
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if 'starvector-8b' in self.model_name:
                if "rotary_emb.inv_freq" in name:
                    continue

                if 'image_encoder' in name:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    continue
                
                for (param_name, weight_name, shard_id) in stacked_params_mapping:

                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    if self.config.tie_word_embeddings and "lm_head.weight" in name:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
            else:
                if "lm_head.weight" in name:
                    continue
                if (
                    ".num_batches_tracked" in name
                    or ".running_mean" in name
                    or ".running_var" in name
                ):
                    # Skip batch norm statistics.
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
