from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
model.config.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

# 예시 preference pair
prompt = "What is the capital of France?"
chosen = "The capital of France is Paris."
rejected = "I think France might be in Canada."

batch = tokenizer(
    [prompt + chosen, prompt + rejected],
    return_tensors="pt",
    padding=True,
)

# log probability from reference model
with torch.no_grad():
    ref_logits = ref_model(**batch).logits[:, :-1]
ref_log_probs = F.log_softmax(ref_logits, dim=-1)
ref_scores = ref_log_probs.gather(2, batch["input_ids"][:, 1:].unsqueeze(-1)).sum(dim=1)

# log probability from model
model_logits = model(**batch).logits[:, :-1]
log_probs = F.log_softmax(model_logits, dim=-1)
scores = log_probs.gather(2, batch["input_ids"][:, 1:].unsqueeze(-1)).sum(dim=1)

# DPO Loss
chosen_score, rejected_score = scores[0], scores[1]
ref_chosen, ref_rejected = ref_scores[0], ref_scores[1]
pi_c = torch.exp(chosen_score - ref_chosen)
pi_r = torch.exp(rejected_score - ref_rejected)
loss = -torch.log(pi_c / (pi_c + pi_r))

# 이 값은 현재 모델이 얼마나 chosen을 선호하지 못하고 있는지를 수치화한 것
# 값이 작을수록 chosen을 잘 예측하고 있다는 의미
print(f"DPO Loss: {loss.item()}")
