import jax
import jax.numpy as jnp
from typing import List, Dict, Any

# log recompilation
jax.config.update("jax_log_compiles", True)


def gelu(x):
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    exp_x = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True))
    return exp_x / jnp.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float = 1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    x = (x - mean) / jnp.sqrt(variance + eps)
    return g * x + b


def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b


def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]
    
    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]
    
    return x


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / jnp.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    
    # split into qkv
    qkv = jnp.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]
    
    # split into heads
    qkv_heads = [jnp.split(x, n_head, axis=-1) for x in qkv]  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]
    
    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - jnp.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]
    
    # perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) 
                for q, k, v in zip(qkv_heads[0], qkv_heads[1], qkv_heads[2])]
    
    # merge heads
    x = jnp.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]
    
    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    
    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    
    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    # token + positional embeddings
    x = wte[inputs] + wpe[jnp.arange(len(inputs))]
    
    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    
    # projection to vocab
    x = layer_norm(x, **ln_f)
    return x @ wte.T


def generate_step(inputs, params, n_head, temperature=1.0):
    logits = gpt2(inputs, **params, n_head=n_head)

    # Apply temperature scaling
    logits = logits / temperature

    # Sample from the distribution
    next_id = logits[-1].argmax()

    return next_id


def generate(inputs: List[int], params: Dict[str, Any], n_head: int, n_tokens_to_generate: int, 
            temperature: float = 1.0):
    from tqdm import tqdm
    
    inputs = jnp.array(inputs)
    generated_ids = []
    
    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        next_id = generate_step(inputs, params, n_head, temperature)
        generated_ids.append(int(next_id))
        inputs = jnp.append(inputs, next_id)
    
    return generated_ids


def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", 
         models_dir: str = "models", temperature: float = 1.0):
    from utils import load_encoder_hparams_and_params
    
    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    
    # Convert numpy arrays to jax arrays
    params = jax.tree_map(jnp.array, params)
    
    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)
    
    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
    
    # generate output ids
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate, temperature)
    
    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)