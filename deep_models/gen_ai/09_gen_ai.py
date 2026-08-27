
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import time

from torch import cuda
import torch

# print("cuda status : ", cuda.is_available())
# print("cuda device name : ", cuda.get_device_name() if cuda.is_available() else "No GPU")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

cache_dir = "./deep_models/gen_ai/cache_dir"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = cache_dir)

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir)

while(True):
    user_input = input("\nEnter the prompt : ")

    prompt = user_input

    if user_input == "stop":
        print("Bye!")
        break

    start = time.time()

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False
    )

    # print("\nShape of outputs : ", outputs.shape)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nTime take to generate response : ", time.time() - start)

    print("\n",response)

"""
Q1. Why does model generates unwanted, lengthy and irrelevant response 
for simple queries like - how many days are there in a week?
A1: Reduce max_new_tokens count to 30. It simply reduces the unwanted length for straight forward question-answers.
A2: Give the model an instruction. The model follows the instructions randomly not always. Not reliable at this point.
A3: Use the model's chat template. Model produces answer in strange manner. It breaks the response in 3 parts:
system-> about model, user-> query, assistant-> actual response to the query
A4: Set generation parameters. Temperature, sample, padding -> does not improve the exaggeration.

Q2. 
"""

#our implementation of model.generate()
def generate(inputs):
    with torch.no_grad():
        for i in range(30):
            output = model(**inputs)
            # print(output.logits.shape)
            logits = output.logits
            next_pred_logits = logits[:, -1, :]
            next_token = torch.argmax(next_pred_logits, dim=-1)
            next_token_unsqueezed = torch.unsqueeze(next_token, dim=-1)

            inputs["input_ids"] = torch.cat((inputs["input_ids"], next_token_unsqueezed), -1)
            inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones(1,1, dtype=torch.int64)), -1) 

            response = tokenizer.decode(next_token_unsqueezed, skip_special_tokens=True)
            print(response[0], end="")

#This method only implements inference logic.
#transformer's model.generate() do:
"""
Forward pass
KV Cache
EOS detection
Temperature
Top-k sampling
Top-p sampling
Beam Search
Repetition penalty
"""

def input_to_prompt(user_input):
    prompt = f"""
    Answer the following question in one concise sentence.

    Question:
    {user_input}

    Answer:
    """
    return prompt

def input_to_chat_template(user_input):
    messages = [
        {"role": "user", "content": user_input}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt