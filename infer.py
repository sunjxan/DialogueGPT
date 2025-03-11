import os
import torch

from data import create_tokenizer
from DialogueGPT import DialogueGPT

def process_data(model, text, tokenizer, device='cpu'):
    """处理输入数据并生成编码器输出"""
    tokens = tokenizer.tokenize(text)[-model.max_seq_len:]
    return torch.LongTensor(tokens).unsqueeze(0).to(device)

def get_probs(model, input_ids, tokenizer, temperature=1.0, top_k=None):
    """获取下一个token的概率分布"""
    input_ids = input_ids[:, -model.max_seq_len:]
    
    pad_token = tokenizer.special_tokens_map['pad_token']
    pad_id = tokenizer.convert_tokens_to_ids(pad_token)
    mask = model.generate_mask(input_ids, pad_id)
    with torch.no_grad():
        output = model(input_ids, mask)
    # 应用温度缩放
    output = output[:, -1] / temperature
    # Top-k 过滤
    if top_k is not None and 0 < top_k <= output.size(-1):
        indices_to_remove = output < torch.topk(output, top_k)[0][..., -1, None]
        output[indices_to_remove] = float('-inf')
    return torch.softmax(output, dim=-1)

def greedy_decode(model, text, tokenizer, max_len=50, device='cpu'):
    model.eval()
    
    input_ids = process_data(model, text, tokenizer, device=device)
    end_token = tokenizer.eos_id()
    
    for _ in range(max_len):
        probs = get_probs(model, input_ids, tokenizer)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == end_token:
            break
    
    return input_ids[0].cpu().tolist()

def sampling_decode(model, text, tokenizer, max_len=50, temperature=1.0, top_k=1, device='cpu'):
    model.eval()
    
    input_ids = process_data(model, text, tokenizer, device=device)
    end_token = tokenizer.eos_id()
    
    for _ in range(max_len):
        probs = get_probs(model, input_ids, tokenizer, temperature, top_k)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == end_token:
            break
    
    return input_ids[0].cpu().tolist()

if __name__ == '__main__':
    tokenizer = create_tokenizer()
    
    # 创建模型
    model = DialogueGPT(tokenizer.vocab_size())
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    ckpt_path = './checkpoints/checkpoint_best.pth'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    text = input('请输入中文句子：\n')
    print('\ninput:', text)
    
    predictions = greedy_decode(model, text, tokenizer, device=device)
    print('\ngreedy decode:', tokenizer.detokenize(predictions))
    
    # 技术文档生成（高确定性）
    predictions = sampling_decode(model, text, tokenizer, max_len=50, temperature=0.7, top_k=3, device=device)
    print('\nsampling decode(高确定性):', tokenizer.detokenize(predictions))
    
    # 创意写作（高多样性）
    predictions = sampling_decode(model, text, tokenizer, max_len=50, temperature=1.2, top_k=8, device=device)
    print('\nsampling decode(高多样性):', tokenizer.detokenize(predictions))
    
    # 平衡模式
    predictions = sampling_decode(model, text, tokenizer, max_len=50, temperature=0.9, top_k=5, device=device)
    print('\nsampling decode(平衡模式):', tokenizer.detokenize(predictions))
