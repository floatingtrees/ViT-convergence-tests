import torch
from biased_vit import ViT
from calflops import calculate_flops


model = ViT(
    image_size = (224, 224),
    patch_size = 14,
    num_classes = 1000,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.0,
    emb_dropout = 0.0
)

input_shape = (1, 3, 224, 224)

flops, macs, params  = calculate_flops(model = model, input_shape = input_shape, output_as_string = True, output_precision = 4)

print(flops, macs, params)