include("lib.jl")

using JSON
using PyCall
using BSON: @save

@pyimport tensorflow as tf
@pyimport torch

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

function get_galactica_tokenizer()
    py"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('facebook/galactica-125m')
    """
    py"tokenizer"
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

function opt_from_pytorch_ckpt(torch_ckpt_path; n_layers, n_head)
    torch_model = torch.load(torch_ckpt_path)

    wpe = permutedims(torch_model["model.decoder.embed_positions.weight"].numpy(), (2, 1)) |> Embedding
    wte = permutedims(torch_model["model.decoder.embed_tokens.weight"].numpy(), (2, 1)) |> Embedding

    blocks = []
    for i in 0:n_layers-1
        attn_weights = permutedims(hcat([
            torch_model["model.decoder.layers.$i.self_attn.q_proj.weight"].numpy()',
            torch_model["model.decoder.layers.$i.self_attn.k_proj.weight"].numpy()',
            torch_model["model.decoder.layers.$i.self_attn.v_proj.weight"].numpy()'
        ]...), (2, 1))
        attn_bias = vcat([
            torch_model["model.decoder.layers.$i.self_attn.q_proj.bias"].numpy(),
            torch_model["model.decoder.layers.$i.self_attn.k_proj.bias"].numpy(),
            torch_model["model.decoder.layers.$i.self_attn.v_proj.bias"].numpy()
        ]...)

        push!(blocks, TransformerDecoderBlock(
            MHA(
                Dense(
                    attn_weights,
                    attn_bias
                ), # attn
                Dense(
                    # permutedims(torch_model["model.decoder.layers.$i.self_attn.out_proj.weight"].numpy(), (2, 1)),
                    torch_model["model.decoder.layers.$i.self_attn.out_proj.weight"].numpy(),
                    torch_model["model.decoder.layers.$i.self_attn.out_proj.bias"].numpy()
                ), # proj
                n_head # nhead
            ),
            FFN(
                Dense(
                    torch_model["model.decoder.layers.$i.fc1.weight"].numpy(),
                    torch_model["model.decoder.layers.$i.fc1.bias"].numpy(),
                ), # fc
                Dense(
                    torch_model["model.decoder.layers.$i.fc2.weight"].numpy(),
                    torch_model["model.decoder.layers.$i.fc2.bias"].numpy(),
                ) # proj
            ),
            LN(
                torch_model["model.decoder.layers.$i.self_attn_layer_norm.weight"].numpy(),
                torch_model["model.decoder.layers.$i.self_attn_layer_norm.bias"].numpy(),
            ),
            LN(
                torch_model["model.decoder.layers.$i.final_layer_norm.weight"].numpy(),
                torch_model["model.decoder.layers.$i.final_layer_norm.bias"].numpy(),
            ),
        ))
    end

    ln_f = LN(
        torch_model["model.decoder.final_layer_norm.weight"].numpy(),
        torch_model["model.decoder.final_layer_norm.bias"].numpy()
    )

    lm_head = Embedding(
        permutedims(torch_model["lm_head.weight"].numpy(), (2, 1))
    )

    OPT(wte, wpe, blocks, ln_f, lm_head)
end
