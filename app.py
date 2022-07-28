from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer
    model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b3", use_cache=True)
    print("hi")
    # conditionally load to GPU
    if device == "cuda:0":
        print("loading to GPU...")
        model.cuda()
        print("done")

    tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b3")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=200,  top_k=1, temperature=0.9, repetition_penalty = 2.0)
    # print(outputs)
    # Decode output tokens
    output_text = tokenizer.decode(outputs[0])
    result = {"output": output_text}

    # Return the results as a dictionary
    return result
