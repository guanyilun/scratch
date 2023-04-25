include("lib.jl")

using JSON
using PyCall
using BSON: @save

@pyimport tensorflow as tf

# hacky way to get encoder from python
function get_encoder()
    py"""
    import sys
    sys.path.insert(0, "/home/yilun/work/sketchbook/nano-llm/")
    import encoder
    enc = encoder.get_encoder('gpt2_124m') # not nice
    """
    return py"enc"
end

function gpt2_from_tf_ckpt(tf_checkpoint_path)
    hparams = JSON.parsefile(tf_checkpoint_path*"/hparams.json")

    wte = Embedding(permutedims(tf.train.load_variable(tf_checkpoint_path, "model/wte"), (2, 1))) # [n_embed, n_vocab]
    wpe = Embedding(permutedims(tf.train.load_variable(tf_checkpoint_path, "model/wpe"), (2, 1))) # [n_embed, ctx_len]

    blocks = []
    for i in 0:hparams["n_layer"]-1
        push!(blocks, TransformerDecoderBlock(
            MHA(
                Dense(
                    permutedims(dropdims(tf.train.load_variable(tf_checkpoint_path, "model/h$i/attn/c_attn/w"), dims=1), (2, 1)),
                    tf.train.load_variable(tf_checkpoint_path, "model/h$i/attn/c_attn/b")
                ), # attn
                Dense(
                    permutedims(dropdims(tf.train.load_variable(tf_checkpoint_path, "model/h$i/attn/c_proj/w"), dims=1), (2, 1)),
                    tf.train.load_variable(tf_checkpoint_path, "model/h$i/attn/c_proj/b")
                ), # proj
                hparams["n_head"]
            ),
            FFN(
                Dense(
                    permutedims(dropdims(tf.train.load_variable(tf_checkpoint_path, "model/h$i/mlp/c_fc/w"), dims=1), (2, 1)),
                    tf.train.load_variable(tf_checkpoint_path, "model/h$i/mlp/c_fc/b")
                ), # fc
                Dense(
                    permutedims(dropdims(tf.train.load_variable(tf_checkpoint_path, "model/h$i/mlp/c_proj/w"), dims=1), (2, 1)),
                    tf.train.load_variable(tf_checkpoint_path, "model/h$i/mlp/c_proj/b")
                )    # proj
            ),
            LN(
                tf.train.load_variable(tf_checkpoint_path, "model/h$i/ln_1/g"),
                tf.train.load_variable(tf_checkpoint_path, "model/h$i/ln_1/b")
            ), 
            LN(
                tf.train.load_variable(tf_checkpoint_path, "model/h$i/ln_2/g"),
                tf.train.load_variable(tf_checkpoint_path, "model/h$i/ln_2/b")
            ), 
        ))
    end
    ln_f = LN(
        tf.train.load_variable(tf_checkpoint_path, "model/ln_f/g"),
        tf.train.load_variable(tf_checkpoint_path, "model/ln_f/b")
    )
    embedding = EmbedTokens(wte, wpe)
    GPT2(embedding, blocks, ln_f)
end