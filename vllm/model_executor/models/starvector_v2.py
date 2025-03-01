from typing import Any, Iterable, List, Optional, Set, Tuple, Union, Mapping
import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import BatchFeature
from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.model_executor.models.gpt_bigcode import GPTBigCodeForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalFieldConfig, MultiModalKwargs)
from vllm.multimodal.parse import (MultiModalDataItems)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.multimodal.parse import (ImageEmbeddingItems, ImageProcessorItems, MultiModalDataItems)
from vllm.model_executor.layers.sampler import get_sampler
from .interfaces import SupportsLoRA, SupportsMultiModal, SupportsPP
from .utils import (WeightsMapper)
from transformers import (AutoTokenizer, AutoConfig, PretrainedConfig)
from transformers.processing_utils import ProcessorMixin
from .vision import get_vision_encoder_info
from functools import cached_property
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode, pad

# TODO add this manually
from starvector.model.image_encoder.image_encoder import ImageEncoder

logger = init_logger(__name__)

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

class StarCoderModel(nn.Module):
    def __init__(self, vllm_config, prefix: str = "", **kwargs):
        super(StarCoderModel, self).__init__()

        self.init_tokenizer(vllm_config.model_config.hf_config.starcoder_model_name)

        self.max_length = vllm_config.model_config.hf_config.max_length
        model_config = AutoConfig.from_pretrained(vllm_config.model_config.hf_config.starcoder_model_name)

        kwargs = {}
        kwargs["trust_remote_code"] = True

        # Configure special tokens for generation
        model_config.eos_token_id = self.tokenizer.eos_token_id
        model_config.pad_token_id = self.tokenizer.pad_token_id
        model_config.bos_token_id = self.tokenizer.bos_token_id
        model_config.vocab_size = len(self.tokenizer)

        vllm_config.model_config.hf_config.n_inner = model_config.n_inner
        vllm_config.model_config.hf_config.layer_norm_epsilon = model_config.layer_norm_epsilon
        vllm_config.model_config.hf_config.max_position_embeddings = model_config.max_position_embeddings
        vllm_config.model_config.hf_config.activation_function = model_config.activation_function
        vllm_config.model_config.hf_config.n_embd = model_config.n_embd

        model = GPTBigCodeForCausalLM(vllm_config=vllm_config, prefix=prefix)
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
    def __init__(self, vllm_config, prefix: str = "", **kwargs):
        super().__init__()
        config = vllm_config.model_config.hf_config

        # Build Code LLM (StarCoder)
        self.svg_transformer = StarCoderModel(vllm_config, prefix=prefix, **kwargs)

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
        if (self.task == "im2svg"):
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
        if len(images.shape) == 5:
            # Collapse second dimension into batch dimension
            B, S, C, H, W = images.shape
            images = images.reshape(B * S, C, H, W)
        # images = torch.randn(31, 1, 3, 224, 224).to('cuda').to(torch.float16)
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
            visual_embeddings = self.embed_image(image_input.to(self.model_precision))
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

class SimpleStarVectorProcessor(ProcessorMixin):
    attributes = ["tokenizer"]  # Only include tokenizer in attributes
    valid_kwargs = ["size", "mean", "std"]  # Add other parameters as valid kwargs
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, 
                 tokenizer=None,  # Make tokenizer the first argument
                 size=224, 
                 mean=None, 
                 std=None, 
                 **kwargs,
                 ):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        # Store these as instance variables
        self.mean = mean
        self.std = std
        self.size = size
        
        self.normalize = transforms.Normalize(mean=mean, std=std)        
        
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode == "RGBA" else img),
            transforms.Lambda(lambda img: self._pad_to_square(img)),
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            self.normalize
        ])

        # Initialize parent class with tokenizer
        super().__init__(tokenizer=tokenizer)


    def __call__(self, images=None, text=None, **kwargs) -> BatchFeature:
        """
        Process images and/or text inputs.
        
        Args:
            images: Optional image input(s)
            text: Optional text input(s)
            **kwargs: Additional arguments
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        image_inputs = {}
        if images is not None:
            if isinstance(images, (list, tuple)):
                images_ = [self.transform(img) for img in images]
            else:
                images_ = self.transform(images)
            image_inputs = {"pixel_values": images_}
        
        text_inputs = {}
        if text is not None:
            text_inputs = self.tokenizer(text, **kwargs)
        return BatchFeature(data={**text_inputs, **image_inputs})

    def _pad_to_square(self, img):
        # Calculate padding to make the image square
        width, height = img.size
        max_dim = max(width, height)
        padding = [(max_dim - width) // 2, (max_dim - height) // 2]
        padding += [max_dim - width - padding[0], max_dim - height - padding[1]]
        return pad(img, padding, fill=255)  # Assuming white padding


class StarVectorProcessingInfo(BaseProcessingInfo):

    def get_vision_encoder_info(self):
        return get_vision_encoder_info(self.get_hf_config())
    
    def get_hf_processor(self):
        return self.ctx.get_hf_processor(SimpleStarVectorProcessor)

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        config = self.ctx.model_config.hf_config
        if config.image_encoder_type == "clip":
            return {"image": 257}
        else:
            return {"image": 576}

    def _apply_feature_select_strategy(
        self,
        strategy: str,
        encoder_num_image_tokens: int,
    ) -> int:
        if strategy == "default":
            return encoder_num_image_tokens - 1
        if strategy == "full":
            return encoder_num_image_tokens

        msg = f"Unexpected feature select strategy: {strategy!r}"
        raise NotImplementedError(msg)

    def get_num_image_tokens(self) -> int:
        hf_config = self.get_hf_config()
        vision_encoder_type = hf_config.image_encoder_type
        if vision_encoder_type == "clip":
            return 257
        else:
            return 576

class StarVectorDummyInputsBuilder(BaseDummyInputsBuilder[StarVectorProcessingInfo]):

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        num_images = mm_counts.get("image", 0)

        processor = self.info.get_hf_processor()
        image_token = '<image-start>' # processor.image_token
        mm_data = {"image": self._get_dummy_images(width=processor.size,
                                   height=processor.size,
                                   num_images=1)[0]}

        return ProcessorInputs(
            prompt_text=image_token * num_images,
            mm_data=mm_data,
        )


class StarVectorMultiModalProcessor(BaseMultiModalProcessor[StarVectorProcessingInfo]):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )
    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:
        hf_config = self.info.get_hf_config()
        image_token_id = 49154

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems))

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                num_image_tokens = self.info.get_num_image_tokens()

            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]

@MULTIMODAL_REGISTRY.register_processor(
    StarVectorMultiModalProcessor,
    info=StarVectorProcessingInfo,
    dummy_inputs=StarVectorDummyInputsBuilder,
)
class StarVectorForCausalLM(nn.Module, SupportsMultiModal, SupportsLoRA, SupportsPP):

    _no_split_modules = []
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    } 
    # LoRA specific attributes
    supported_lora_modules = [r'model\.svg_transformer\.transformer\.model\.layers\.\d+\.(self_attn\.(o_proj|qkv_proj)|mlp\.(c_fc|c_proj))']
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = []

    # check if this is needed
    # Map HuggingFace weight prefixes to vLLM ones.
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
    })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs):
        super().__init__()
        config: StarVectorConfig = vllm_config.model_config.hf_config
        self.model_name = config._name_or_path

        if 'starvector-1b' in self.model_name:
            self.model = StarVectorModel(vllm_config=vllm_config, prefix=prefix, **kwargs)
        else:   
            from vllm.model_executor.models.starvector2 import StarVector2Model
            self.model = StarVector2Model(config=vllm_config, prefix=prefix, **kwargs)


    # @cached_property
    # def sampler(self):
    #     if hasattr(self.model, "sampler"):
    #         return self.model.sampler

    #     return get_sampler()
    

    def _parse_and_validate_image_input(self, **kwargs: object) -> Optional[dict]:
        pixel_values = kwargs.pop("pixel_values", None)
        return pixel_values
    
    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata,  # typically an AttentionMetadata object
                intermediate_tensors: Optional[Any] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                pixel_values: torch.Tensor = None,
                **kwargs: object
                ) -> Union[torch.Tensor, Any]:
        
        return self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            **kwargs,
        )
    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: Any) -> Optional[torch.Tensor]:
        return self.model.svg_transformer.transformer.compute_logits(hidden_states, sampling_metadata)

    def sample(self, logits: torch.Tensor, sampling_metadata: Any) -> Optional[Any]:
        return self.model.svg_transformer.transformer.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        # Define the mapping for stacked parameters.
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        for name, loaded_weight in weights:
            # Special treatment for starvector-8b models.
            if "starvector-8b" in self.model_name:
                if "rotary_emb.inv_freq" in name:
                    continue
                if "image_encoder" in name:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue
                found = False
                for (param_name, weight_name, shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    new_name = name.replace(weight_name, param_name)
                    if new_name in params_dict:
                        param = params_dict[new_name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, shard_id)
                        loaded_params.add(new_name)
                        found = True
                        break
                if not found:
                    if getattr(self.config, "tie_word_embeddings", False) and "lm_head.weight" in name:
                        continue
                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)
            else:
                # For non-8b models, skip lm_head and BatchNorm / running stats.
                if "lm_head.weight" in name:
                    continue
                if (".num_batches_tracked" in name or ".running_mean" in name or ".running_var" in name):
                    continue
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
        return loaded_params
