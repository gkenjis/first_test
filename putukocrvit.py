import math

import torch,torch.nn as nn,torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed, VisionTransformer

class ConvModule(nn.Module):
    def __init__(self,d): 
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(d,d*2,1),
            nn.GLU(1),
            nn.Conv1d(d,d,31,padding=15,groups=d),
            nn.BatchNorm1d(d),nn.SiLU(),
            nn.Conv1d(d,d,1)
            )
    def forward(self,x): 
        return self.net(x.transpose(1,2)).transpose(1,2)

class FeedForward(nn.Module):
    def __init__(self,d,h): 
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,h),nn.SiLU(),nn.Dropout(0.1),nn.Linear(h,d))
    def forward(self,x): return self.net(x)

class ConformerBlock(nn.Module):
    def __init__(self, d=256, h=4, ff=1024):
        super().__init__()
        self.ff1 = FeedForward(d, ff)
        self.ff2 = FeedForward(d, ff)
        self.attn = nn.MultiheadAttention(d,h,batch_first=True)
        self.conv = ConvModule(d)

        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.n3 = nn.LayerNorm(d)
        self.n4 = nn.LayerNorm(d)

    def forward(self, x):
        x = x + 0.5 * self.ff1(self.n1(x))

        y = self.n2(x)
        y, _ = self.attn(y,y,y,need_weights=False)
        x = x + y

        x = x + self.conv(self.n3(x))
        x = x + 0.5 * self.ff2(self.n4(x))

        return x

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self,d_model,max_len=2048):
        super().__init__()
        self.pe=nn.Parameter(torch.randn(1,max_len,d_model)*0.02)

    def forward(self,x):
        return x+self.pe[:,:x.size(1)]


class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=2048,dropout=0.1):
        super().__init__()

        pos=torch.arange(max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

        pe=torch.zeros(1,max_len,d_model)
        pe[0,:,0::2]=torch.sin(pos*div)
        pe[0,:,1::2]=torch.cos(pos*div)

        self.register_buffer("pe",pe)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.dropout(x+self.pe[:,:x.size(1)])
    

class Encoder(VisionTransformer):
    def __init__(self, img_size=(48,768),patch_size=(4,4),in_chans=3,embed_dim=256,depth=12,num_heads=8,mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
    ):
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            embed_layer=embed_layer,
            num_classes=0, 
            global_pool='', 
            class_token=False,
        )

    def forward(self, x):
        # Return all tokens
        return self.forward_features(x)
    
class OCRModel(nn.Module):
    def __init__(self,num_classes=180,IMG_H=48,IMG_W=768):
        super().__init__()
        d_model=256
        layers=6
        head = 4
        patch_size = (4,4)
        self.out_h = IMG_H // patch_size[0]
        self.out_w = IMG_W // patch_size[1]

        self.backbone=Encoder(img_size=(IMG_H,IMG_W), patch_size=patch_size, embed_dim=d_model,depth=layers)
        self.proj=nn.Linear(d_model*self.out_h,d_model)
        self.pos_emb=PositionalEncoding(d_model=d_model)
        self.encoder=nn.Sequential(*[ConformerBlock(d_model,head,d_model*3) for _ in range(layers)])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,num_classes)

    def forward(self,x):
        x=self.backbone(x)
        B = x.size(0)
        D = x.size(2)
        x = x.reshape(B, self.out_h, self.out_w, D)
        B,H,W,C = x.shape
        x = x.permute(0,2,1,3)
        x = x.reshape(B,W,C*H)
        x=self.proj(x)
        x=self.pos_emb(x)
        x=self.encoder(x)
        x=self.encoder_norm(x)
        return self.head(x)

class SimpleCTCLoss(nn.Module):
    """
    パディング済み2次元targetsに対応するCTC Loss。
    logits:
        [B, T, C]
    targets:
        [B, S]
    target_lengths:
        [B]
    target_lengthsより後ろのtargetsはパディングとして無視される。
    blank_index=0の場合、パディング値にも0を使用できる。
    """

    def __init__(self, blank_index: int = 0) -> None:
        super().__init__()

        if blank_index < 0:
            raise ValueError("blank_indexは0以上である必要があります。")

        self.blank_index = blank_index
        self.loss_func = nn.CTCLoss(blank=blank_index, reduction="mean", zero_infinity=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, num_classes = logits.shape
        targets = targets.to(device=logits.device, dtype=torch.long)
        target_lengths = target_lengths.to(device="cpu", dtype=torch.long)
        input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
        log_probs = logits.float().log_softmax(dim=2).permute(1, 0, 2).contiguous()
        return self.loss_func(log_probs, targets, input_lengths, target_lengths)


@torch.no_grad()
def decode_ctc(logits, tokenizer, blank_index=0):
    token_ids = logits.argmax(dim=-1).cpu()

    texts = []

    for sequence in token_ids:
        sequence = torch.unique_consecutive(sequence)
        sequence = sequence[sequence != blank_index]

        texts.append(tokenizer.decode(sequence.tolist()))

    return texts

def print_parameter_count(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'    Total: {total_num/1000000:.4f} M, Trainable: {trainable_num/1000000:.4f} M')

if __name__ == "__main__":
    num_classes = 180
    IMG_H=32
    IMG_W=768
    model = OCRModel(num_classes=num_classes, IMG_H=IMG_H,IMG_W=IMG_W)
    criterion = SimpleCTCLoss(blank_index=0)
    print_parameter_count(model)

    images = torch.randn(4, 3, IMG_H, IMG_W)
    logits = model(images)
    targets = torch.tensor([
        [1, 2, 3],
        [4, 5, 0],
        [6, 0, 0],
        [7, 8, 9],
    ], dtype=torch.long)

    target_lengths = torch.tensor([3, 2, 1, 3], dtype=torch.long)
    loss = criterion(logits, targets, target_lengths)
    print(f"Input Shape(Image): {images.shape}")
    print(f"Output Shape(Logits): {logits.shape}")
    print(loss.item())
    # decode_ctc(logits)
