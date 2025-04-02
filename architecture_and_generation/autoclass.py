from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

model_name = "meta-llama/Llama-3.2-1B"

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Config Class:", type(config))
print("Tokenizer Class:", type(tokenizer))
print("Model Class:", type(model))
