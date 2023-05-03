#%%
from rwkv_basic import rwkv_net
from rwkv_utils import get_tokenizer, parse_rwkv_weight, parse_model_info, rnn_generate

#%%
tokenizer = get_tokenizer()
weights_tree = parse_rwkv_weight()
model_info = parse_model_info(weights_tree)

#%%
prompt = "The quick brown fox jumps over the lazy dog"
out = rnn_generate(rwkv_net, weights_tree, prompt, tokenizer=tokenizer)

