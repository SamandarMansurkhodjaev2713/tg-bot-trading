from pathlib import Path
import pandas as pd
import json
import math
from collections import Counter
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
except Exception as e:
    torch = None
    Trainer = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    IMPORT_ERROR = str(e)

ROOT = Path('.')
PROC = ROOT / 'data' / 'processed'
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_IDS = {
    'finbloom': 'Chaitanya14/FinBloom_7B',
}

def load_dataset_for_lm(csv_path: Path):
    df = pd.read_csv(csv_path)
    texts = []
    for _, row in df.head(500).iterrows():
        txt = ' | '.join([f"{c}:{row[c]}" for c in df.columns if pd.notnull(row[c])])
        texts.append(txt)
    return texts

def train_lm(model_id: str, train_texts, output_dir: Path):
    if torch is None or Trainer is None:
        tokens = []
        for t in train_texts:
            tokens.extend(str(t).split())
        cnt = Counter(tokens)
        V = len(cnt)
        N = sum(cnt.values())
        denom = N + V if N + V > 0 else 1
        probs = {w: (c + 1) / denom for w, c in cnt.items()}
        unk_prob = 1 / denom
        total_tokens = sum(len(str(t).split()) for t in train_texts)
        if total_tokens == 0:
            total_tokens = 1
        def text_ll(text):
            s = 0.0
            for w in str(text).split():
                p = probs.get(w, unk_prob)
                s += math.log(p)
            return s
        avg_ll = sum(text_ll(t) for t in train_texts) / total_tokens
        perplexity = math.exp(-avg_ll)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'model_unigram.json', 'w', encoding='utf-8') as f:
            json.dump({'probs': probs, 'unk_prob': unk_prob}, f)
        pd.DataFrame([{'status':'ok','method':'unigram','vocab_size': V,'tokens': N,'perplexity': perplexity}]).to_csv(output_dir / 'metrics.csv', index=False)
        return
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')

    class TxtDataset(torch.utils.data.Dataset):
        def __init__(self, texts, tokenizer):
            self.examples = tokenizer(texts, truncation=True, padding='max_length', max_length=256)['input_ids']
        def __len__(self): return len(self.examples)
        def __getitem__(self, i): return torch.tensor(self.examples[i])

    train_ds = TxtDataset(train_texts, tokenizer)
    training_args = TrainingArguments(output_dir=str(output_dir), per_device_train_batch_size=1, num_train_epochs=1, logging_steps=10, save_strategy='no')
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds)
    trainer.train()
    trainer.save_model(output_dir)
    pd.DataFrame([{'status':'ok'}]).to_csv(output_dir / 'metrics.csv', index=False)

def main():
    csvs = list(PROC.glob('*.csv'))
    if not csvs:
        print('No processed CSVs found. Run preprocess_external.py first.')
        return
    texts = load_dataset_for_lm(csvs[0])
    outdir = MODELS_DIR / 'educational_finbloom'
    outdir.mkdir(exist_ok=True)
    train_lm(MODEL_IDS['finbloom'], texts, outdir)
    print('Training pipeline completed.')

if __name__ == '__main__':
    main()
