from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor

class FinBloom7BAnalyst:
    def __init__(self):
        self.model = GradientBoostingRegressor()
    def fit(self, X, y):
        self.model.fit(X, y)
    def predict(self, X):
        return self.model.predict(X)
    def save(self, path):
        dump(self.model, path)

def try_load_financial_agent():
    try:
        from peft import PeftModel, PeftConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        peft_model_id = "Chaitanya14/Financial_Agent"
        config = PeftConfig.from_pretrained(peft_model_id)
        tok = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base, peft_model_id)
        return tok, model
    except Exception:
        return None, None
