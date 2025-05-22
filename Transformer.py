import math
import torch
import torch.nn as nn

from Decoder import Decoder

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model=768, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.position_emb = nn.Embedding(max_seq_len, d_model)
        self.role_emb = nn.Embedding(3, d_model)  # 0:系统 1:用户 2:机器人
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, role_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        token_emb = self.token_emb(input_ids)
        position_emb = self.position_emb(position_ids)
        role_emb = self.role_emb(role_ids)
        embeddings = token_emb + position_emb + role_emb
        return self.dropout(embeddings)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, d_ff=3072, max_seq_len=512, dropout=0.1):
        """
        Transformer 模型
        Args:
            vocab_size (int): 词表大小
            d_model (int): 模型维度（输入/输出维度）
            num_heads (int): 多头注意力头数
            num_layers (int): Decoder 层数
            d_ff (int): 前馈网络中间层维度
            max_seq_len (int): 最大序列长度
            dropout (float): Dropout 概率
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        
        # 1. 词嵌入层 + 位置编码 + 角色编码
        self.embed = nn.Embedding(vocab_size, d_model, max_seq_len, dropout)
        
        # 2. 解码器
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        # 3. 最终线性层
        self.generator = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重绑定：输入嵌入和输出层共享权重
        self.generator.weight = self.embed.token_emb.weight
    
    def forward(self, input_ids, role_ids, mask=None):
        """
        前向传播
        Args:
            input_ids (Tensor): 序列 (batch_size, seq_len)
            mask (Tensor): 序列掩码 (batch_size, seq_len, seq_len)
        Returns:
            output (Tensor): 输出概率分布 (batch_size, seq_len, vocab_size)
        """
        seq_len = input_ids.size(-1)
        assert seq_len <= self.max_seq_len, f"序列长度{seq_len}超过最大限制{self.max_seq_len}"
        
        # 1. 词嵌入 + 位置编码 + 角色编码
        emb = self.embed(input_ids)  # (batch_size, seq_len, d_model)
        
        # 2. 解码器处理
        decoder_output = self.decoder(emb, mask)  # (batch_size, seq_len, d_model)
        
        # 3. 输出层映射到词表
        output = self.generator(decoder_output)  # (batch_size, seq_len, vocab_size)
        
        return output
    
    def init_parameters(self):
        """
        初始化模型参数
        """
        for name, param in self.named_parameters():
            if param.dim() > 1:  # 仅初始化矩阵权重，忽略偏置和LayerNorm参数
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:  # 偏置初始化为零
                nn.init.zeros_(param)
            # LayerNorm参数保持默认初始化（gamma=1, beta=0）
    
    @staticmethod
    def generate_padding_mask(seq, pad_id=0):
        """生成填充掩码（pad位置为False）"""
        return (seq != pad_id).unsqueeze(-2)  # (batch_size, 1, seq_len)
    
    @staticmethod
    def generate_causal_mask(seq_len):
        """生成因果掩码（下三角为True）"""
        return torch.tril(torch.ones(seq_len, seq_len)) == 1  # (seq_len, seq_len)
    
    @classmethod
    def generate_mixed_mask(cls, seq, pad_id=0):
        '''结合填充掩码和因果掩码得到目标序列掩码'''
        return cls.generate_padding_mask(seq, pad_id) & cls.generate_causal_mask(seq.size(-1)).to(seq.device)   # (batch_size, seq_len, seq_len)

'''
    计算模型参数量
    
    1. 嵌入层
    vocab_size × d_model
    
    2. 位置编码
    max_seq_len × d_model
    
    3. 角色编码
    3 x d_model
    
    4. 解码器（Decoder）
    每层包含：
        1个多头注意力（相当于4个线性层）：4 × (d_model × d_model + d_model)
        前馈网络（2个线性层）：2 × d_model × d_ff + d_ff + d_model
        2个归一化层：2 × (d_model + d_model)
    最终归一化层：d_model + d_model
    总参数量：
        num_decoder_layers × [4d² + 2d·d_ff + d_ff + 9d] + 2d
    
    5. 生成器（Generator）
    weight共享嵌入层权重，无bias
'''
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
