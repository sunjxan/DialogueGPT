import os
import re
import json
import torch

from zhconv import convert
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

corpus_file = 'cleaned_corpus.txt'
model_dirname = 'tokenizer_chinese'

def clean_text(text):
    # 定义正则表达式模式匹配对话块
    pattern = r'<\|start_header_id\|>(.*?)<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>'
    
    # 查找所有匹配项
    matches = re.findall(pattern, text, re.DOTALL)
    
    # 初始化结果列表和临时存储
    result = []
    last_role = last_content = None
    
    for role, content in matches:
        content = convert(content.strip(), 'zh-cn')
        
        if last_role == role:
            last_content += '\n\n' + content
        else:
            if last_role:
                result.append({"role": last_role, "content": last_content})
            last_role, last_content = role, content
    
    if last_role:
        result.append({"role": last_role, "content": last_content})       
    return result

if not os.path.exists(corpus_file):
    
    dataset = load_dataset('neo-lin/chat_alpaca_chinese_llama_3.1', split='train')
    
    # 应用清洗并保存
    with open(corpus_file, "w", encoding="utf-8") as f:
        for example in dataset:
            cleaned = clean_text(example["text"])
            if cleaned:
                f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

if not os.path.exists(model_dirname):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    tokenizer.save_pretrained(model_dirname)

def create_tokenizer():
    return AutoTokenizer.from_pretrained(model_dirname)

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = tokenizer.encode(line.strip()) + [tokenizer.eos_id()]
                for i in range(0, len(tokens), max_len):
                    text = tokens[i:i+max_len]
                    if len(text) > 1:
                        self.data.append(text)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_batch(batch, tokenizer):
    data = []
    
    for item in batch:
        data.append(torch.LongTensor(item))
    
    return pad_sequence(data, batch_first=True, padding_value=tokenizer.pad_id())

def create_dataloader(tokenizer, batch_size, max_len=512, shuffle=False, drop_last=False):
    dataset = TextDataset(corpus_file, tokenizer, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
        collate_fn=lambda batch: collate_batch(batch, tokenizer))
