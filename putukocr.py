import math

import torch,torch.nn as nn,torch.nn.functional as F

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
    def __init__(self,d=256,h=4,ff=1024):
        super().__init__()
        self.ff1=FeedForward(d,ff); self.ff2=FeedForward(d,ff)
        self.attn=nn.MultiheadAttention(d,h,batch_first=True)
        self.conv=ConvModule(d)
        self.n1,self.n2,self.n3,self.n4=[nn.LayerNorm(d) for _ in range(4)]

    def forward(self,x):
        x=x+0.5*self.ff1(self.n1(x))
        a,_=self.attn(self.n2(x),self.n2(x),self.n2(x))
        x=x+a
        x=x+self.conv(self.n3(x))
        x=x+0.5*self.ff2(self.n4(x))
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
    
class SEModule(nn.Module):
    def __init__(self,c,r=4): 
        super().__init__() 
        self.avg=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Conv2d(c,c//r,1),nn.SiLU(),nn.Conv2d(c//r,c,1),nn.Sigmoid())

    def forward(self,x): 
        return x*self.fc(self.avg(x))

class DSConv(nn.Module):

    def __init__(self,i,o,s=(1,1),k=5,se=True,residual=True):
        super().__init__()

        self.use_residual = residual and s==(1,1) and i==o

        self.dw = nn.Sequential(
            nn.Conv2d(i,i,kernel_size=k,stride=s,padding=k//2,groups=i,bias=False),
            nn.BatchNorm2d(i),
            nn.SiLU()
        )

        self.pw = nn.Sequential(
            nn.Conv2d(i,o,1,bias=False),
            nn.BatchNorm2d(o),
            nn.SiLU()
        )

        self.se = SEModule(o) if se else nn.Identity()

    def forward(self,x):
        y=self.se(self.pw(self.dw(x)))
        return x+y if self.use_residual else y

class LCNetV4HSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(
            nn.Conv2d(3,32,3,stride=(2,1),padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        self.blocks=nn.Sequential(
            DSConv(32,64,s=(2,1),k=3),
            DSConv(64,128,s=(2,2),k=3),
            DSConv(128,256,s=(2,1),k=3),
            DSConv(256,512,s=(1,2),k=3),
            DSConv(512,512,s=(1,1),k=3),            
        )
    def forward(self,x): 
        return self.blocks(self.stem(x))


class LCNetV4H(nn.Module):

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=(2,1),padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        self.stage1 = nn.Sequential(
            DSConv(32,64,s=(2,1),k=3),
            DSConv(64,64,s=(1,1),k=5)
        )

        self.stage2 = nn.Sequential(
            DSConv(64,128,s=(2,2),k=5),
            DSConv(128,128,s=(1,1),k=5),
            DSConv(128,128,s=(1,1),k=5)
        )

        self.stage3 = nn.Sequential(
            DSConv(128,256,s=(2,1),k=5),
            DSConv(256,256,s=(1,1),k=7),
            DSConv(256,256,s=(1,1),k=7),
            DSConv(256,256,s=(1,1),k=7)
        )

        self.stage4 = nn.Sequential(
            DSConv(256,512,s=(1,2),k=7),
            DSConv(512,512,s=(1,1),k=7),
            DSConv(512,512,s=(1,1),k=9),
            DSConv(512,512,s=(1,1),k=9)
        )

    def forward(self,x):
        x=self.stem(x)
        x=self.stage1(x)
        x=self.stage2(x)
        x=self.stage3(x)
        x=self.stage4(x)
        return x
    
class OCRModel(nn.Module):
    def __init__(self,num_classes=18710,IMG_H=48):
        super().__init__()
        d_model=384
        layers=6
        head = 4
        self.backbone=LCNetV4H()
        self.proj=nn.Linear(int(512*IMG_H/16),d_model)
        self.pos_emb=PositionalEncoding(d_model=d_model)
        self.encoder=nn.Sequential(*[ConformerBlock(d_model,head,d_model*3) for _ in range(layers)])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,num_classes)

    def forward(self,x):
        x=self.backbone(x)
        B,C,H,W = x.shape
        x = x.permute(0,3,1,2).reshape(B,W,C*H)
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
def decode_ctc(logits: torch.Tensor, tokenizer=None, blank_index: int = 0) -> list[str]:
    token_ids = logits.argmax(dim=-1)  # [B, T]
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
    IMG_H=48
    model = OCRModel(num_classes=num_classes, IMG_H=IMG_H)
    criterion = SimpleCTCLoss(blank_index=0)
    print_parameter_count(model)

    images = torch.randn(4, 3, IMG_H, 768)
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
