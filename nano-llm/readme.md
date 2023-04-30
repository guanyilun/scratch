# transformer
Learn and experimenting with transformer by reimplementing
a minimum version of gpt-2 in julia following the implementation
in [picoGPT](https://github.com/jaymody/picoGPT/).

## Attempt 1
In my first attempt, I will implement most of the
building blocks from scratch following picoGPT. I avoided using
existing library such as Flux or NNlib. 
- [x] Learned how to work with column major convention in julia
- [x] Understood the basic building blocks
- [x] Generate outputs that are consistent with picoGPT
- [x] Realise some trickness in array row / column conversion from python to julia

## Attempt 2 
Ideas for the second attempt
- [x] use more building blocks from Flux and NNlib
- [x] able to load pretrained weights
- [x] able to work with new weights
- [x] generate correct output
- [x] running model on gpu
It doesn't look like I can get training working with this version. I will
work on training in attempt 3

## Attempt 3
Ideas
- [X] batched input
- [X] use batch_mul
- [X] get gradient working
- [X] data loaders
- [X] basic training
- [X] wandb integration
- [X] longer period training on gpu seems fine too, the memory usage is a bit abnormally high, need to investigate
Try more robust dataloading, DataLoader.jl, MLUtils have some useful.

## Attempt 4
Ideas
- [X] experiment with chains
- [X] position embedding
- [X] checkpointing
- [X] multiple gpus (using distributed)
- [X] data streaming to multiple gpus
- [ ] multiple gpus (using mpi)
- [X] tested galactica / opt causaul model
- [X] tested tuning
- [ ] dropout support for training
- [X] tested rwkv model implementation

## Attempt 5
Ideas
- slightly more professional rwkv implementation
