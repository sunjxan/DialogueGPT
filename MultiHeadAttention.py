import math
import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        """
        缩放点积注意力机制
        Args:
            dropout (float): Dropout概率，默认为0.1
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # Dropout层
    
    def forward(self, Q, K, V, mask=None):
        """
        前向传播
        Args:
            Q: 查询张量, shape (batch_size, num_heads, seq_len_q, d_k)
            K: 键张量, shape (batch_size, num_heads, seq_len_k, d_k)
            V: 值张量, shape (batch_size, num_heads, seq_len_k, d_v)
            mask: 掩码张量, shape (batch_size, 1, seq_len_q, seq_len_k)
        
        Returns:
            注意力输出: shape (batch_size, num_heads, seq_len_q, d_v)
            注意力权重: shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        # 计算Q和K的点积得分
        scores = torch.matmul(Q, K.transpose(-2, -1))  # Q·K^T
        # scores shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # 缩放操作：除以sqrt(d_k)防止梯度消失
        d_k = K.size(-1)  # 获取K的最后一个维度d_k
        scores = scores / math.sqrt(d_k)
        # scores shape保持不变: (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # 应用掩码（如果需要）
        if mask is not None:
            # 将mask中为False的位置替换为负无穷大（softmax后趋近于0）
            scores = scores.masked_fill(mask == 0, float('-inf'))
            # mask需要能广播到scores的形状
            # mask形状(T, T)，右侧和右上方为False，用于重新编码时忽略pad和该词后面的词
        
        # 计算注意力权重（最后一维进行softmax）
        attn_weights = torch.softmax(scores, dim=-1)
        # attn_weights shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        
        attn_weights = self.dropout(attn_weights)
        
        # 将注意力权重应用到V上
        output = torch.matmul(attn_weights, V)
        # output shape: (batch_size, num_heads, seq_len_q, d_v)
        
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        多头注意力机制
        Args:
            d_model: 输入维度（总维度）
            num_heads: 注意力头的数量
            dropout: dropout概率
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 线性变换矩阵
        self.W_qkv = nn.Linear(d_model, 3 * d_model)  # (d_model, 3*d_model) 合并QKV投影
        self.W_o = nn.Linear(d_model, d_model)  # (d_model, d_model)
        
        self.attn = ScaledDotProductAttention(dropout)
    
    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x: QKV向量 (batch_size, seq_len, d_model)
            mask: 掩码 (batch_size, seq_len, seq_len)
        Returns:
            输出: (batch_size, seq_len, d_model)
            注意力权重: (batch_size, num_heads, seq_len, seq_len)
        """    
        batch_size = x.size(0)
        
        # 线性变换 + 分割多头
        QKV = self.W_qkv(x)  # (batch_size, seq_len, 3*d_model)
        Q, K, V = QKV.chunk(3, dim=-1)  # 各(batch_size, seq_len, d_model)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k)  # (batch_size, seq_len, num_heads, d_k)
        K = K.view(batch_size, -1, self.num_heads, self.d_k)  # (batch_size, seq_len, num_heads, d_k)
        V = V.view(batch_size, -1, self.num_heads, self.d_k)  # (batch_size, seq_len, num_heads, d_k)
        
        # 转置维度以便矩阵计算 (batch_size, num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2).contiguous()
        K = K.transpose(1, 2).contiguous()
        V = V.transpose(1, 2).contiguous()
        
        # 应用掩码（如果存在）
        if mask is not None:
            # 扩展掩码维度以匹配多头 (batch_size, 1, seq_len, seq_len) -> 广播到num_heads
            mask = mask.unsqueeze(1)
        
        output = self.attn(Q, K, V, mask)
        
        # 转置回维度 (batch_size, seq_len, num_heads, d_k)
        output = output.transpose(1, 2).contiguous()
        
        # 拼接所有头 (batch_size, seq_len, d_model)
        output = output.view(batch_size, -1, self.d_model)
        
        # 最终线性变换
        output = self.W_o(output)
        
        return output
