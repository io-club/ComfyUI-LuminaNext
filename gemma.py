import torch
import sys
import os
import comfy
from comfy import sd1_clip
from comfy.ldm.modules.attention import optimized_attention_for_device
from transformers import GemmaTokenizerFast

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class GemmaRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float, dtype=None, device=None, operations=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(dim, dtype=dtype, device=device))

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (comfy.ops.cast_to_input(self.weight, x) + 1.0) * x

class GemmaMLP(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype=None, device=None, operations=None):
        super().__init__()
        self.down_proj = operations.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.gate_proj = operations.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.up_proj = operations.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype, device=device)
        self.act_fn = lambda a: torch.nn.functional.gelu(a, approximate="tanh")

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class GemmaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, dtype=None, device=None):
        super().__init__()
        self.dtype = dtype
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    def forward(self, x, qlen):
        position_ids = torch.arange(qlen, device=x.device, dtype=self.dtype).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin

class GemmaAttention(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 head_dim, num_heads, num_key_value_heads,
                 max_position_embeddings, rope_theta,
                 dtype, device, operations):
        super().__init__()
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.q_proj = operations.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False, dtype=dtype, device=device)
        self.k_proj = operations.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, dtype=dtype, device=device)
        self.v_proj = operations.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, dtype=dtype, device=device)
        self.o_proj = operations.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False, dtype=dtype, device=device)
        self.rotary_emb = GemmaRotaryEmbedding(self.head_dim, self.max_position_embeddings, self.rope_theta, dtype=dtype, device=device)

    def forward(self, hidden_states, attention_mask, optimized_attention):
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(v, q_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        is_causal = True if attention_mask is None and q_len > 1 else False
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=is_causal)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        return self.o_proj(attn_output)

class GemmaDecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int,
                 head_dim: int, num_heads: int, num_key_value_heads: int,
                 eps: float,
                 max_position_embeddings: int, rope_theta: float,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.input_layernorm = GemmaRMSNorm(hidden_size, eps, dtype, device, operations)
        self.mlp = GemmaMLP(hidden_size, intermediate_size, dtype, device, operations)
        self.post_attention_layernorm = GemmaRMSNorm(hidden_size, eps, dtype, device, operations)
        self.self_attn = GemmaAttention(hidden_size,
                                        head_dim, num_heads, num_key_value_heads,
                                        max_position_embeddings, rope_theta,
                                        dtype, device, operations)

    def forward(self, hidden_states, attention_mask=None, optimized_attention=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            optimized_attention=optimized_attention,
        )
        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return self.mlp(hidden_states) + residual

class Gemma(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        self.num_layers = config_dict["num_hidden_layers"]
        self.hidden_size = config_dict["hidden_size"]
        eps = config_dict["rms_norm_eps"]
        intermediate_size = config_dict["intermediate_size"]
        head_dim = config_dict["head_dim"]
        num_heads = config_dict["num_attention_heads"]
        num_key_value_heads = config_dict["num_key_value_heads"]
        max_position_embeddings = config_dict["max_position_embeddings"]
        rope_theta = config_dict["rope_theta"]
        self.dtype = dtype
        self.embed_tokens = operations.Embedding(config_dict["vocab_size"], self.hidden_size, device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList(
            [GemmaDecoderLayer(self.hidden_size, intermediate_size,
                               head_dim, num_heads, num_key_value_heads,
                               eps,
                               max_position_embeddings, rope_theta,
                               dtype, device, operations) for _ in range(self.num_layers)]
        )
        self.norm = GemmaRMSNorm(self.hidden_size, eps, dtype, device, operations)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, embeddings):
        self.embed_tokens = embeddings

    def forward(self, input_ids, attention_mask=None, intermediate_output=None, final_layer_norm_intermediate=True, dtype=None):
        x = self.embed_tokens(input_ids, out_dtype=dtype)
        normalizer = torch.tensor(self.hidden_size**0.5, dtype=x.dtype)
        x = x * normalizer
        intermediate = None
        optimized_attention = optimized_attention_for_device(x.device, mask=attention_mask is not None, small_input=True)
        for i, l in enumerate(self.layers):
            x = l(x, attention_mask, optimized_attention)
            if i == intermediate_output:
                intermediate = x.clone()
        x = self.norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.norm(intermediate)
        return x, intermediate

class LuminaGemmaClip(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", layer="last", layer_idx=None, dtype=None, model_options={}):
        textmodel_json_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gemma", "config.json")
        super().__init__(device=device, layer=layer, layer_idx=layer_idx, textmodel_json_config=textmodel_json_config, dtype=dtype, special_tokens={"end": 1, "pad": 0}, model_class=Gemma, model_options=model_options)
        self.dtypes = set()
        if dtype is not None:
            self.dtypes.add(dtype)

class GemmaTokenizerFixed(GemmaTokenizerFast):
    @classmethod
    def from_pretrained(cls, path: str):
        return super().from_pretrained(path, add_eos_token=True)

class LuminaGemmaTokenizer(sd1_clip.SDTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gemma")
        super().__init__(tokenizer_path, embedding_directory=embedding_directory, embedding_size=2048, embedding_key='gemma', tokenizer_class=GemmaTokenizerFixed, has_start_token=True, pad_token=0, pad_to_max_length=False, max_length=999999)

    def tokenize_with_weights(self, text:str, return_word_ids=False):
        batched_tokens = super().tokenize_with_weights(text, True)
        for batch in batched_tokens:
            padlen = ((len(batch) // 8) + 1) * 8 - len(batch)
            batch.extend([(self.pad_token, 1.0, 0)] * padlen)
        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w, _ in x] for x in batched_tokens]
        return batched_tokens
