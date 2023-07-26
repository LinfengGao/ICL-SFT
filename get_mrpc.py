import datasets

task = "Natural Language Inference"
queries = ""
label_map = {0: "Equivalent.", 1: "Not equivalent."}

data = datasets.load_dataset("glue", name="mrpc", cache_dir="/home/glf/data/cache", split="train")

for i in range(10):
    with open("./template/MRPC_template", "r", encoding="utf-8") as mrpc_template_file:
        mrpc_content = mrpc_template_file.read().format(
            sentence1=data[i]["sentence1"],
            sentence2=data[i]["sentence2"],
            label=label_map[data[i]["label"]]
        )
        queries += mrpc_content + "\n"

with open("./template/ICL_template", "r", encoding="utf-8") as icl_template_file:
    icl_content = icl_template_file.read().format(
        task=task,
        queries=queries
    )

with open("./instruction_generation/mrpc_generation", 'w', encoding='utf-8') as f:
    f.write(icl_content)
