"""
dense_model.py — v7.2.6 export copy
Router-free K=0 Dense model. Identical to v7.2.6/dense_model.py except
TaskConditionedSharedExpert uses jnp.take for gamma/beta indexing so that
direction_ids (a JAX tracer during jax2tf) does not trigger __array__().
"""
import jax
import jax.numpy as jnp
from flax import nnx
from config import MoEModelConfig


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return jnp.concatenate((-x2, x1), axis=-1)


class MultiHeadAttention(nnx.Module):
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float, dtype: any, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nnx.Linear(d_model, d_model, use_bias=False, dtype=dtype, rngs=rngs)
        self.key   = nnx.Linear(d_model, d_model, use_bias=False, dtype=dtype, rngs=rngs)
        self.value = nnx.Linear(d_model, d_model, use_bias=False, dtype=dtype, rngs=rngs)
        self.out   = nnx.Linear(d_model, d_model, use_bias=False, dtype=dtype, rngs=rngs)

        self.q_norm = nnx.RMSNorm(self.head_dim, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)
        self.k_norm = nnx.RMSNorm(self.head_dim, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)

        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x, mask=None, context=None, sin=None, cos=None, deterministic: bool = False):
        batch_size, seq_len, _ = x.shape
        context = x if context is None else context
        ctx_len = context.shape[1]

        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.key(context).reshape(batch_size, ctx_len, self.num_heads, self.head_dim)
        v = self.value(context).reshape(batch_size, ctx_len, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if sin is not None and cos is not None:
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)

        attn_scores = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(self.head_dim)

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            attn_scores = jnp.where(mask == 1, attn_scores, jnp.array(-jnp.inf, dtype=attn_scores.dtype))

        attn_probs = jax.nn.softmax(attn_scores.astype(jnp.float32), axis=-1).astype(attn_scores.dtype)
        attn_probs = self.dropout(attn_probs, deterministic=deterministic)

        attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_probs, v)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        return self.out(attn_output)


class TaskConditionedSharedExpert(nnx.Module):
    def __init__(self, d_model: int, mlp_dim: int, num_tasks: int, dropout_rate: float, dtype: any, rngs: nnx.Rngs):
        self.w1 = nnx.Linear(d_model, mlp_dim, use_bias=False, dtype=dtype, rngs=rngs)
        self.w2 = nnx.Linear(d_model, mlp_dim, use_bias=False, dtype=dtype, rngs=rngs)
        self.w3 = nnx.Linear(mlp_dim, d_model, use_bias=False, dtype=dtype, rngs=rngs)

        self.gamma = nnx.Param(jnp.zeros((num_tasks, mlp_dim), dtype=jnp.float32))
        self.beta  = nnx.Param(jnp.zeros((num_tasks, mlp_dim), dtype=jnp.float32))
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x, direction_ids, deterministic: bool = False):
        # jnp.take instead of numpy indexing so direction_ids can be a traced
        # value during jax2tf export (numpy[tracer] calls __array__() and fails).
        # mode='clip': direction_ids is always 0 or 1, so clip is a safe no-op.
        g_full = jnp.take(jnp.asarray(self.gamma.value), direction_ids, axis=0, mode='clip')
        b_full = jnp.take(jnp.asarray(self.beta.value),  direction_ids, axis=0, mode='clip')
        g = g_full[:, None, :] if x.ndim == 3 else g_full
        b = b_full[:, None, :] if x.ndim == 3 else b_full

        h = jax.nn.silu(self.w1(x)) * self.w2(x)
        h = h * (1.0 + g) + b
        h = self.dropout(h, deterministic=deterministic)

        # Scale factor is mathematically baked into w3 during extraction.
        return self.w3(h)


class DenseLayer(nnx.Module):
    def __init__(self, d_model: int, mlp_dim: int, dropout_rate: float, dtype: any, rngs: nnx.Rngs):
        self.shared_expert = TaskConditionedSharedExpert(d_model, mlp_dim, 2, dropout_rate, dtype, rngs)

    def __call__(self, x, direction_ids, deterministic: bool = False):
        out = self.shared_expert(x, direction_ids, deterministic=deterministic)
        return out, 0.0, 0.0


class EncoderBlock(nnx.Module):
    def __init__(self, cfg: MoEModelConfig, rngs: nnx.Rngs):
        self.ln_1 = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=jnp.float32, rngs=rngs)
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout_rate, cfg.dtype, rngs)
        self.ln_2 = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=jnp.float32, rngs=rngs)
        self.moe = DenseLayer(cfg.d_model, cfg.mlp_dim, cfg.dropout_rate, cfg.dtype, rngs)
        self.res_dropout = nnx.Dropout(cfg.dropout_rate, rngs=rngs)

    def __call__(self, x, mask, direction_ids, sin=None, cos=None, deterministic=False):
        attn_out = self.self_attn(self.ln_1(x), mask=mask, sin=sin, cos=cos, deterministic=deterministic)
        x = x + self.res_dropout(attn_out, deterministic=deterministic)

        moe_out, aux_loss, z_loss = self.moe(self.ln_2(x), direction_ids, deterministic=deterministic)
        x = x + self.res_dropout(moe_out, deterministic=deterministic)
        return x, aux_loss, z_loss


class DecoderBlock(nnx.Module):
    def __init__(self, cfg: MoEModelConfig, rngs: nnx.Rngs):
        self.ln_1 = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=jnp.float32, rngs=rngs)
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout_rate, cfg.dtype, rngs)
        self.ln_2 = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=jnp.float32, rngs=rngs)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout_rate, cfg.dtype, rngs)
        self.ln_3 = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=jnp.float32, rngs=rngs)
        self.moe = DenseLayer(cfg.d_model, cfg.mlp_dim, cfg.dropout_rate, cfg.dtype, rngs)
        self.res_dropout = nnx.Dropout(cfg.dropout_rate, rngs=rngs)

    def __call__(self, x, tgt_mask, enc_out, src_mask, direction_ids, sin=None, cos=None, deterministic=False):
        attn_out = self.self_attn(self.ln_1(x), mask=tgt_mask, sin=sin, cos=cos, deterministic=deterministic)
        x = x + self.res_dropout(attn_out, deterministic=deterministic)

        cross_out = self.cross_attn(self.ln_2(x), mask=src_mask, context=enc_out, deterministic=deterministic)
        x = x + self.res_dropout(cross_out, deterministic=deterministic)

        moe_out, aux_loss, z_loss = self.moe(self.ln_3(x), direction_ids, deterministic=deterministic)
        x = x + self.res_dropout(moe_out, deterministic=deterministic)
        return x, aux_loss, z_loss


class Encoder(nnx.Module):
    def __init__(self, cfg: MoEModelConfig, rngs: nnx.Rngs):
        self.blocks = nnx.List([EncoderBlock(cfg, rngs) for _ in range(cfg.num_layers)])
        self.ln_final = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=jnp.float32, rngs=rngs)

    def __call__(self, x, mask, direction_ids, sin=None, cos=None, deterministic=False):
        total_aux = 0.0
        total_z = 0.0
        for block in self.blocks:
            x, aux, z = block(x, mask, direction_ids, sin=sin, cos=cos, deterministic=deterministic)
            total_aux += aux
            total_z += z
        return self.ln_final(x), total_aux, total_z


class Decoder(nnx.Module):
    def __init__(self, cfg: MoEModelConfig, rngs: nnx.Rngs):
        self.blocks = nnx.List([DecoderBlock(cfg, rngs) for _ in range(cfg.num_layers)])
        self.ln_final = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=jnp.float32, rngs=rngs)

    def __call__(self, x, tgt_mask, enc_out, src_mask, direction_ids, sin=None, cos=None, deterministic=False):
        total_aux = 0.0
        total_z = 0.0
        for block in self.blocks:
            x, aux, z = block(x, tgt_mask, enc_out, src_mask, direction_ids, sin=sin, cos=cos, deterministic=deterministic)
            total_aux += aux
            total_z += z
        return self.ln_final(x), total_aux, total_z


class DenseTranslationModel(nnx.Module):
    def __init__(self, cfg: MoEModelConfig, rngs: nnx.Rngs):
        self.cfg = cfg
        self.embedding = nnx.Embed(cfg.vocab_size, cfg.d_model, dtype=cfg.dtype, rngs=rngs)
        self.embed_norm = nnx.RMSNorm(cfg.d_model, dtype=cfg.dtype, param_dtype=jnp.float32, rngs=rngs)

        self.encoder = Encoder(cfg, rngs)
        self.decoder = Decoder(cfg, rngs)

    def _get_rope(self, positions):
        head_dim = self.cfg.d_model // self.cfg.num_heads
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        freqs = jnp.einsum('i,j->ij', positions.astype(jnp.float32), inv_freq)
        emb = jnp.concatenate((freqs, freqs), axis=-1)

        sin = jnp.sin(emb)[None, :, None, :].astype(self.cfg.dtype)
        cos = jnp.cos(emb)[None, :, None, :].astype(self.cfg.dtype)
        return sin, cos

    def encode(self, source_ids, src_mask, deterministic=False):
        seq_len = source_ids.shape[1]
        positions = jnp.arange(seq_len)
        direction_ids = (source_ids[:, 0] == self.cfg.vi_en_token_id).astype(jnp.int32)

        sin, cos = self._get_rope(positions)

        x = self.embed_norm(self.embedding(source_ids))
        enc_out, _, _ = self.encoder(x, src_mask, direction_ids, sin=sin, cos=cos, deterministic=deterministic)
        return enc_out

    @property
    def decoder_blocks(self):
        return self.decoder.blocks
