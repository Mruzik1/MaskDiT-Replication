from torch import nn
import torch.nn.functional as F
import torch


class AttentionBlock(nn.Module):
    def __init__(
            self, 
            dim, 
            num_heads, 
            qkv_bias=True, 
            attn_dropout=0.,
            out_drop=0.,
            fused_attn=True
        ):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        
        self.layer_norm = nn.LayerNorm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.out_drop = nn.Dropout(out_drop)
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.out_linear = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv_values = self.qkv(x).reshape(3, b, self.num_heads, n, self.head_dim)
        q, k, v = qkv_values.unbind(0)
        q = self.layer_norm(q)
        k = self.layer_norm(k)

        # use built-in attention calculation
        if self.fused_attn:
             x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.out_linear(x)
        x = self.out_drop(x)
        return x if self.fused_attn else (x, attn)


if __name__ == "__main__":
    torch.manual_seed(42)
    seq_length = 10
    feature_dim = 64
    batch_size = 1
    num_heads = 8
    input_tensor = torch.rand(batch_size, seq_length, feature_dim).to("cuda")

    attention_block = AttentionBlock(
        feature_dim, 
        num_heads, 
        qkv_bias=True, 
        fused_attn=True
    ).to("cuda")

    output = attention_block(input_tensor)
    print(output.shape)