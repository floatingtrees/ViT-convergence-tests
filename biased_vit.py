import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.transforms as transforms

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            attn = self.attend(dots + mask)
        else:
            attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, mask):
        for i, (attn, ff) in enumerate(self.layers):
            if i == len(self.layers) - 1:
                mask = None
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.patch_height = patch_height
        self.channels = channels
        self.patch_width = patch_width
        self.image_height = image_height
        self.image_width = image_width

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pre_to_patch_embedding = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width)
        self.unpatchify = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', w = image_width // patch_width, h = image_height // patch_height, p1 = self.patch_height, p2 = self.patch_width)
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)


    def generate_attention_mask(self, sequence_length, mask_constant, device = "cpu"):
        mask = torch.zeros((1, sequence_length, sequence_length), device = device)
        mask.fill_(mask_constant)
        indices = torch.arange(sequence_length)
        num_offset_height_patches = self.image_width // self.patch_width
        mask[0, indices, indices] = 0
        mask[0, indices[:-1] + 1, indices[:-1]] = 0 # attends to the patch on the left
        mask[0, indices[1:] - 1, indices[1:]] = 0 # attends to the patch on the right
        mask[0, indices[num_offset_height_patches:] - num_offset_height_patches, indices[num_offset_height_patches:]] = 0 # attends to the one on top
        mask[0, indices[num_offset_height_patches - 1:] - num_offset_height_patches + 1, indices[num_offset_height_patches - 1:]] = 0 # top right
        mask[0, indices[num_offset_height_patches + 1:] - num_offset_height_patches - 1, indices[num_offset_height_patches + 1:]] = 0
        mask[0, indices[:-num_offset_height_patches] + num_offset_height_patches, indices[:-num_offset_height_patches]] = 0 # attend to the bottom token
        mask[0, indices[:-num_offset_height_patches + 1] + num_offset_height_patches - 1, indices[:-num_offset_height_patches + 1]] = 0 # bottom left
        mask[0, indices[:-num_offset_height_patches - 1] + num_offset_height_patches + 1, indices[:-num_offset_height_patches - 1]] = 0 # bottom right
        return mask



    def forward(self, img, mask_constant = float('-inf')): # mask constant is what we set the mask to
        b, c, h, w = img.shape
        if h != self.image_height or w != self.image_width or c != self.channels:
            raise AssertionError("Height, width, or num_channels does not match")
        img = self.pre_to_patch_embedding(img)
        batch_size, sequence_length, features = img.shape
        mask = self.generate_attention_mask(sequence_length + 1, mask_constant)

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask = mask)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)