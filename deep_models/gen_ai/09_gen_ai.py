
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import time

from torch import cuda

# print("cuda status : ", cuda.is_available())
# print("cuda device name : ", cuda.get_device_name() if cuda.is_available() else "No GPU")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

cache_dir = "./deep_models/gen_ai/cache_dir"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = cache_dir)

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = cache_dir)

while(True):
    user_input = input("\nEnter the prompt : ")

    prompt = f"""
    Answer the following question in one concise sentence.

    Question:
    {user_input}

    Answer:
    """

    # messages = [
    #     {"role": "user", "content": user_input}
    # ]

    # prompt = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )


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

    print("\nShape of outputs : ", outputs.shape)

    print("\nOutput : ", outputs)

    print("\nOutput [0] : ", outputs[0])

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