import logging
import os
from pathlib import PosixPath

import tiktoken
import torch
import torch.nn as nn
from sentencepiece import SentencePieceProcessor

from .transformer import CausalTransformer, TransformerConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------
# Loading GPT-2
# --------------------------------------------------------------------------------


class GPT2(nn.Module):
    """
    Load GPT2 model

    Parameters
    ----------
    model: str
        name of the model, either "small", "medium", "large", "xl"
    save_dir: PosixPath, optional
        path to directory to save and load GPT-2 weights
    """

    def __init__(self, model, save_dir=None):
        super().__init__()

        # build the architecture from the CausalTransformer class
        config_args = {
            "small": dict(emb_dim=768, n_head=12, n_layer=12),  # 124M params
            "medium": dict(emb_dim=1024, n_head=16, n_layer=24),  # 350M params
            "large": dict(emb_dim=1280, n_head=20, n_layer=36),  # 774M params
            "xl": dict(emb_dim=1600, n_head=25, n_layer=48),  # 1558M params
        }[model]
        config_args = config_args | dict(
            vocab_size=50_257,
            pos_emb=True,
            seq_len=1024,
            attn_bias=True,
            ffn_bias=True,
            norm_bias=True,
            activation="gelu",
            norm="layer",
            pre_norm=True,
        )
        config = TransformerConfig(**config_args)
        self.model = CausalTransformer(config)

        # bind forward method to the inner GPT model
        setattr(self, "forward", self.model.forward)

        logger.info(f"Loading GPT2-{model} tokenizer from Tiktoken")
        self.tokenizer = tiktoken.get_encoding("gpt2")

        # try to load weights from save_path
        if save_dir is not None:
            assert isinstance(
                save_dir, PosixPath
            ), "save_dir must be a PosixPath object"
            save_path = save_dir / f"gpt2-{model}.pt"
            if os.path.exists(save_path):
                logger.info(f"Loading GPT2 model from {save_path}")
                self.model.load_state_dict(torch.load(save_path))
                return

        # otherwise load from HuggingFace
        logger.info(f"Loading GPT2-{model} model from HuggingFace Transformers library")
        self.model_name = "gpt2-" + model.lower() if model != "small" else "gpt2"
        self.emb_dim = config.emb_dim
        self.n_layer = config.n_layer
        self.load_from_huggingface()

        if save_dir is not None:
            logger.info(f"Saving GPT2-{model} model to {save_path}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.model.state_dict(), save_path)

    def load_from_huggingface(self):
        """
        Load weights into the model from HuggingFace Transformers library.

        Parameters
        ----------
        model: str
            name of the model, either gpt2, gpt2-medium, gpt2-large, gpt2-xl
        """
        from transformers import GPT2LMHeadModel

        gpt_state_dict = GPT2LMHeadModel.from_pretrained(self.model_name).state_dict()
        local_state_dict = self.model.state_dict()

        correspondence = {
            "embeddings.token_emb.weight": "transformer.wte.weight",
            "embeddings.pos_emb.weight": "transformer.wpe.weight",
            "output.weight": "lm_head.weight",
            "output_norm.weight": "transformer.ln_f.weight",
            "output_norm.bias": "transformer.ln_f.bias",
        }
        transposed = []
        special = {}
        for layer in range(self.n_layer):
            correspondence = correspondence | {
                f"blocks.{layer}.norm_1.weight": f"transformer.h.{layer}.ln_1.weight",
                f"blocks.{layer}.norm_1.bias": f"transformer.h.{layer}.ln_1.bias",
                f"blocks.{layer}.attn.output.weight": f"transformer.h.{layer}.attn.c_proj.weight",
                f"blocks.{layer}.attn.output.bias": f"transformer.h.{layer}.attn.c_proj.bias",
                f"blocks.{layer}.norm_2.weight": f"transformer.h.{layer}.ln_2.weight",
                f"blocks.{layer}.norm_2.bias": f"transformer.h.{layer}.ln_2.bias",
                f"blocks.{layer}.ffn.fc1.weight": f"transformer.h.{layer}.mlp.c_fc.weight",
                f"blocks.{layer}.ffn.fc1.bias": f"transformer.h.{layer}.mlp.c_fc.bias",
                f"blocks.{layer}.ffn.fc2.weight": f"transformer.h.{layer}.mlp.c_proj.weight",
                f"blocks.{layer}.ffn.fc2.bias": f"transformer.h.{layer}.mlp.c_proj.bias",
            }
            transposed = transposed + [
                f"transformer.h.{layer}.attn.c_attn.weight",
                f"transformer.h.{layer}.attn.c_proj.weight",
                f"transformer.h.{layer}.mlp.c_fc.weight",
                f"transformer.h.{layer}.mlp.c_proj.weight",
            ]
            special = special | {
                f"blocks.{layer}.attn.query.weight": f"transformer.h.{layer}.attn.c_attn.weight",
                f"blocks.{layer}.attn.query.bias": f"transformer.h.{layer}.attn.c_attn.bias",
                f"blocks.{layer}.attn.key.weight": f"transformer.h.{layer}.attn.c_attn.weight",
                f"blocks.{layer}.attn.key.bias": f"transformer.h.{layer}.attn.c_attn.bias",
                f"blocks.{layer}.attn.value.weight": f"transformer.h.{layer}.attn.c_attn.weight",
                f"blocks.{layer}.attn.value.bias": f"transformer.h.{layer}.attn.c_attn.bias",
            }
        for k in correspondence:
            if correspondence[k] in transposed:
                local_state_dict[k] = gpt_state_dict[correspondence[k]].T
            else:
                local_state_dict[k] = gpt_state_dict[correspondence[k]]
        for k in special:
            if "query" in k:
                if "bias" in k:
                    local_state_dict[k] = gpt_state_dict[special[k]][: self.emb_dim]
                else:
                    local_state_dict[k] = gpt_state_dict[special[k]][
                        :, : self.emb_dim
                    ].T
            elif "key" in k:
                if "bias" in k:
                    local_state_dict[k] = gpt_state_dict[special[k]][
                        self.emb_dim : 2 * self.emb_dim
                    ]
                else:
                    local_state_dict[k] = gpt_state_dict[special[k]][
                        :, self.emb_dim : 2 * self.emb_dim
                    ].T
            elif "value" in k:
                if "bias" in k:
                    local_state_dict[k] = gpt_state_dict[special[k]][2 * self.emb_dim :]
                else:
                    local_state_dict[k] = gpt_state_dict[special[k]][
                        :, 2 * self.emb_dim :
                    ].T
        self.model.load_state_dict(local_state_dict)


# --------------------------------------------------------------------------------
# Loading Mistral 7B
# --------------------------------------------------------------------------------


class Mistral(nn.Module):
    """
    Load Mistral-7B model

    Parameters
    ----------
    save_dir: PosixPath, optional
        path to directory to load Mistral original weights
    """

    def __init__(self, save_dir=None):
        super().__init__()

        # build the architecture from the CausalTransformer class
        config_args = dict(
            vocab_size=32_000,
            emb_dim=4096,
            pos_emb=False,
            seq_len=4096,
            n_head=32,
            attn_bias=False,
            attn_downsampling=4,
            rope=True,
            rope_theta=10_000,
            activation="swiglu",
            ffn_dim=14336,
            ffn_bias=False,
            norm="rms",
            norm_bias=False,
            pre_norm=True,
            n_layer=32,
            norm_eps=1e-3,  # this is useful for half precision (Mistral's implementation is 1e-5)
        )

        config = TransformerConfig(**config_args)
        self.model = CausalTransformer(config)

        # bind forward method to the inner GPT model
        setattr(self, "forward", self.model.forward)

        # load weights from mistral checkpoint
        logger.info("Loading Mistral model from mistral checkpoint")
        self.save_dir = save_dir / "mistral-7B-v0.1"
        self.n_layer = config.n_layer
        self.load_from_mistral()

    def load_from_mistral(self):
        """
        Load weights into the model from HuggingFace Transformers library.

        Parameters
        ----------
        model: str
            name of the model, either gpt2, gpt2-medium, gpt2-large, gpt2-xl
        """

        correspondence = {
            "embeddings.token_emb.weight": "tok_embeddings.weight",
            "output_norm.weight": "norm.weight",
            "output.weight": "output.weight",
        }
        for layer in range(self.n_layer):
            correspondence = correspondence | {
                f"blocks.{layer}.norm_1.weight": f"layers.{layer}.attention_norm.weight",
                f"blocks.{layer}.attn.query.weight": f"layers.{layer}.attention.wq.weight",
                f"blocks.{layer}.attn.key.weight": f"layers.{layer}.attention.wk.weight",
                f"blocks.{layer}.attn.value.weight": f"layers.{layer}.attention.wv.weight",
                f"blocks.{layer}.attn.output.weight": f"layers.{layer}.attention.wo.weight",
                f"blocks.{layer}.norm_2.weight": f"layers.{layer}.ffn_norm.weight",
                f"blocks.{layer}.ffn.fc1.weight": f"layers.{layer}.feed_forward.w1.weight",
                f"blocks.{layer}.ffn.fc2.weight": f"layers.{layer}.feed_forward.w2.weight",
                f"blocks.{layer}.ffn.swiglu_mat.weight": f"layers.{layer}.feed_forward.w3.weight",
            }

        local_state_dict = self.model.state_dict()
        try:
            mistral_state_dict = torch.load(self.save_dir / "consolidated.00.pth")
        except:
            raise ValueError(
                f"Could not load Mistral weights from {self.save_dir}.\nYou can download them online."
            )
        for k in local_state_dict:
            if k in correspondence:
                local_state_dict[k] = mistral_state_dict[correspondence[k]]

        self.model.load_state_dict(local_state_dict)

        logger.info("Loading Mistral tokenizer from mistral checkpoint")
        self.tokenizer = SentencePieceProcessor(
            model_file=str(self.save_dir / "tokenizer.model")
        )
