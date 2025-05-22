import os
import torch

from data import create_tokenizer
from Transformer import Transformer

ROLE_MAP = {"system": 0, "user": 1, "assistant": 2}

def process_data(input_ids, role_ids, model, role, tokens, device='cpu'):
    input_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    role_tensor = torch.LongTensor([ROLE_MAP[role]] * len(tokens)).unsqueeze(0).to(device)
    
    input_ids = torch.cat([input_ids, input_tensor], dim=-1)[:, -model.max_seq_len:]
    role_ids = torch.cat([role_ids, role_tensor], dim=-1)[:, -model.max_seq_len:]
    
    return input_ids, role_ids

def get_probs(model, input_ids, role_ids, tokenizer, temperature=1.0, top_k=None):
    input_ids = input_ids[:, -model.max_seq_len:]
    role_ids = role_ids[:, -model.max_seq_len:]
    mask = model.generate_mixed_mask(input_ids, tokenizer.pad_token_id)
    
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            role_ids=role_ids,
            mask=mask
        )
    
    # 应用温度缩放
    output = output[:, -1] / temperature
    # Top-k 过滤
    if top_k is not None and 0 < top_k <= output.size(-1):
        indices_to_remove = output < torch.topk(output, top_k)[0][..., -1, None]
        output[indices_to_remove] = float('-inf')
    return torch.softmax(output, dim=-1)

def sampling_decode(model, input_ids, role_ids, tokenizer, max_len=100, temperature=1.0, top_k=1):
    model.eval()
    
    assistant_id = torch.LongTensor([ROLE_MAP['assistant']]).unsqueeze(0).to(device)
    result = []
    
    for _ in range(max_len):
        probs = get_probs(model, input_ids, role_ids, tokenizer, temperature, top_k)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        role_ids = torch.cat([role_ids, assistant_id], dim=-1)
        result.append(next_token.item())
        if next_token.item() == tokenizer.sep_token_id:
            break
    if not result or result[-1] != tokenizer.sep_token_id:
        result.append(tokenizer.sep_token_id)
    
    return result

if __name__ == '__main__':
    # 设置随机种子（保证可重复性）
    torch.manual_seed(0)
    
    tokenizer = create_tokenizer()
    
    # 创建模型
    model = Transformer(tokenizer.vocab_size)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    ckpt_path = './checkpoints/checkpoint_best.pth'
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    input_ids = torch.LongTensor().unsqueeze(0).to(device)
    role_ids = torch.LongTensor().unsqueeze(0).to(device)
    
    while True:
        
        while True:
            try:
                text = input('>>> ').strip()
                print()
            except:
                print()
                exit()
            
            if text:
                break
        
        if text == '/exit':
            break
        
        if text == '/clear':
            input_ids = torch.LongTensor().unsqueeze(0).to(device)
            role_ids = torch.LongTensor().unsqueeze(0).to(device)
            print('历史已清除')
            print()
            continue
        
        tokens = tokenizer.encode(text, add_special_tokens=False)
        tokens.append(tokenizer.sep_token_id)
        
        input_ids, role_ids = process_data(input_ids, role_ids, model, 'user', tokens, device=device)
        
        predictions = sampling_decode(model, input_ids, role_ids, tokenizer, max_len=100, temperature=0.9, top_k=5)
        input_ids, role_ids = process_data(input_ids, role_ids, model, 'assistant', predictions, device=device)
        
        print(tokenizer.decode(predictions, skip_special_tokens=True).replace(" ", ""))
        print()
