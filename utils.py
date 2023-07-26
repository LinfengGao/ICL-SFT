import torch
from transformers import GenerationConfig

def result_cal(model, tokenizer, text, temperature=0.1, top_p=0.75, top_k=20, num_beams=2, max_new_tokens=260, **kwargs):
    with torch.no_grad(): 
        inputs = tokenizer(text, padding = True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        # 计算反事实和事实词（就是fact-context长度）的长度
        # padding 方向在左边，即[0,id1,id2,...]
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
    
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            max_new_tokens=max_new_tokens,
        )
        len_seq = generation_output.sequences.size()[0]
        output = tokenizer.batch_decode(generation_output.sequences)
        return output, len_seq