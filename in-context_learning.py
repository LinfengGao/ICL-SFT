from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from utils import result_cal
import torch

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1", cache_dir="/home/glf/data/cache").to(device)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1", cache_dir="/home/glf/data/cache")

with open("./instruction_generate_prompt/mrpc_prompt", "r", encoding="utf-8") as f:
    prompt = f.read()
    response, _ = result_cal(model, tokenizer, prompt, max_new_tokens=500)
    with open("./instruction_output/mrpc_instructions", "w", encoding="utf-8") as f:
        f.write(response[0])
