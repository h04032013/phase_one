import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

# Settings
model_name = "Qwen/Qwen2.5-0.5B"
input_path = "/Users/haylindiaz/Projects/Phase_One_Testing/MATH_test_short.json"
cache_str = "/Users/haylindiaz/Projects/Phase_One_Testing"
output_path = "/Users/haylindiaz/Projects/Phase_One_Testing/phase_one_responses.json"
batch_size = 8  # Adjust based on your GPU memory

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_str, padding_side='left',trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_str,trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load input questions
with open(input_path, "r") as f:
    problems = json.load(f)

# Prompt template
prompt_template = (
    "Provide insightful explanations to grade-school students for these math questions"
    "Then, clearly state the final numerical answer on a new line starting with 'Final answer:'.\n\nProblem: {}\n"
)

results = []

# Batch processing
for i in tqdm(range(0, len(problems), batch_size)):
    batch = problems[i:i+batch_size]
    prompts = [prompt_template.format(entry["problem"], ) for entry in batch]

    # Tokenize batch
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)


    print("generating for ", i)
    # Generate responses
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            do_sample=False
        )

    # Decode each response and parse final answer
    for j, output_ids in enumerate(outputs):
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        prompt_len = len(prompts[j])
        response_text = output_text[len(prompts[j]):].strip()

        final_answer = None
        for line in response_text.split("\n"):
            if line.lower().startswith("final answer:"):
                final_answer = line.split(":", 1)[-1].strip()
                break

        results.append({
            "unique_id":  batch[j]["unique_id"],
            "question": batch[j]["problem"],
            "response": response_text,
            "final_answer": final_answer
        })

# Save results
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
