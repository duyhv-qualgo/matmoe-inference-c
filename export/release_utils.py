from pathlib import Path

import jax
import jax.numpy as jnp
import msgpack
from flax import nnx
from flax.serialization import _msgpack_ext_unpack
from transformers import AutoTokenizer

# Backward compatibility for older flax.nnx builds.
if not hasattr(nnx, "List"):
    nnx.List = list

from config import config as base_config, MoEModelConfig
from moe_model import MoETranslationModel


def _default_release_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_tokenizer(tokenizer_dir: str | None = None):
    release_root = _default_release_root()
    tok_dir = Path(tokenizer_dir) if tokenizer_dir else release_root / "tokenizer"
    return AutoTokenizer.from_pretrained(tok_dir)


def build_model_for_dim(tokenizer, mlp_dim: int):
    model_cfg = MoEModelConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        vi_en_token_id=tokenizer.convert_tokens_to_ids("<translate-vi-en>"),
        d_model=base_config.d_model,
        num_heads=base_config.num_heads,
        mlp_dim=mlp_dim,
        num_layers=base_config.num_layers,
        num_experts=base_config.num_experts,
        top_k=base_config.top_k,
        semantic_dim=base_config.semantic_dim,
        dropout_rate=0.0,
        max_seq_len=base_config.max_length_inference,
        dtype=jnp.bfloat16,
    )
    return MoETranslationModel(model_cfg, rngs=nnx.Rngs(base_config.seed))


def load_msgpack_model(model, model_path: str):
    model_path = Path(model_path)
    with open(model_path, "rb") as f:
        msgpack_bytes = f.read()

    saved_dict = msgpack.unpackb(
        msgpack_bytes, ext_hook=_msgpack_ext_unpack, raw=False, strict_map_key=False
    )
    _, current_params, _ = nnx.split(model, nnx.Param, ...)

    def wrap_state(template, raw_dict):
        if hasattr(template, "items"):
            res = {}
            for k, v in template.items():
                val = raw_dict.get(k)
                if val is None and isinstance(k, str) and k.isdigit():
                    val = raw_dict.get(int(k))
                if val is None and isinstance(k, int):
                    val = raw_dict.get(str(k))
                if val is not None:
                    res[k] = wrap_state(v, val)
            return nnx.State(res)
        if isinstance(template, nnx.Variable):
            return type(template)(raw_dict)
        return raw_dict

    restored_params = wrap_state(current_params, saved_dict)
    nnx.update(model, restored_params)
    return model


def count_params(model) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model))))
