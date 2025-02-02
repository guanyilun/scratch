import jax.numpy as np
from jax.lax import rsqrt
from jax import jit
from jax.nn import softmax, sigmoid

def silu(x):
    return x * sigmoid(x)

def swiglu(x, w1, w2):
    return silu(x @ w1) * (x @ w2)

def rms_norm(x, weight, eps: float = 1e-6):
    mean_square = np.mean(x * x, axis=-1, keepdims=True)
    return weight * x * rsqrt(mean_square + eps)

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def rotate_half(x):
    # Assumes the last dimension is even
    x1, x2 = np.split(x, 2, axis=-1)
    return np.concatenate((-x2, x1), axis=-1)

def apply_rotary_pos_emb(q, k, pos):
    cos = np.cos(pos)
    sin = np.sin(pos)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed

def split_heads(x, num_splits):
    batch_size, seq_len, dim = x.shape
    return x.reshape(batch_size, seq_len, num_splits, -1).transpose(0, 2, 1, 3)

def grouped_query_attention(x, pos_emb, w_q, w_k, w_v, num_heads, num_groups, head_dim):
    batch_size, seq_len, _ = x.shape

    # Project Q, K, V
    q = x @ w_q   # Expected shape: (batch_size, seq_len, num_heads * head_dim)
    k = x @ w_k   # Expected shape: (batch_size, seq_len, num_groups * head_dim)
    v = x @ w_v   # Expected shape: (batch_size, seq_len, num_groups * head_dim)

    # Split heads: Q into num_heads, K and V into num_groups
    q = split_heads(q, num_heads)  # Shape: (batch_size, num_heads, seq_len, head_dim)
    k = split_heads(k, num_groups) # Shape: (batch_size, num_groups, seq_len, head_dim)
    v = split_heads(v, num_groups) # Shape: (batch_size, num_groups, seq_len, head_dim)

    # Apply rotary embeddings (ensure pos_emb is broadcastable with q/k)
    q, k = apply_rotary_pos_emb(q, k, pos_emb)

    # Expand grouped keys/values to match query heads via broadcasting
    heads_per_group = num_heads // num_groups
    # Expand dimensions to avoid copies via broadcasting
    k = k[:, :, None, :, :]  # Shape: (batch_size, num_groups, 1, seq_len, head_dim)
    v = v[:, :, None, :, :]
    # Broadcast to (batch_size, num_groups, heads_per_group, seq_len, head_dim)
    k = np.broadcast_to(k, (batch_size, num_groups, heads_per_group, seq_len, head_dim))
    v = np.broadcast_to(v, (batch_size, num_groups, heads_per_group, seq_len, head_dim))
    # Merge num_groups and heads_per_group dimensions into num_heads
    k = k.reshape(batch_size, num_heads, seq_len, head_dim)
    v = v.reshape(batch_size, num_heads, seq_len, head_dim)

    # Compute attention scores: shape (batch_size, num_heads, seq_len, seq_len)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)

    # Create a causal mask (lower triangular matrix) and apply it to the scores.
    # The mask has shape (1, 1, seq_len, seq_len) and is broadcasted over batch and heads.
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))[None, None, :, :]
    scores = np.where(causal_mask, scores, -1e9)

    # Apply softmax over the last dimension (keys dimension)
    attention = softmax(scores, axis=-1)
    out = np.matmul(attention, v)

    # Combine heads: transpose and reshape back to (batch_size, seq_len, num_heads * head_dim)
    out = out.transpose(0, 2, 1, 3)
    return out.reshape(batch_size, seq_len, -1)

def qwen_block(x, pos_emb, block_params):
    # Pre-normalization and attention
    x_norm = rms_norm(x, **block_params['norm1'])
    att_out = grouped_query_attention(x_norm, pos_emb, **block_params['attention'])
    x = x + att_out
    
    # Pre-normalization and FFN with SwiGLU
    x_norm = rms_norm(x, **block_params['norm2'])
    ffn_out = swiglu(x_norm, **block_params['ffn'])
    return x + ffn_out

def qwen_net(tokens, emb, blocks, ln_out, head):
    x = emb['weight'][tokens]
    
    # Generate positional embeddings
    seq_len = x.shape[1]
    head_dim = blocks[0]['attention']['head_dim']
    pos = np.arange(seq_len)[None, :, None] * np.exp(
        -2 * np.arange(head_dim // 2) / head_dim
    )[None, None, :]
    
    # Initial layer norm is stored in block 0
    x = rms_norm(x, **blocks[0]['norm0'])
    
    for i in range(len(blocks)):
        x = qwen_block(x, pos, blocks[i])
        
    x = rms_norm(x, **ln_out)
    logits = head['weight'] @ x
    
    return logits