from transformers import BloomTokenizer, BloomForCausalLM
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer
    model = BloomForCausalLM.from_pretrained("bigscience/bloom-1b3")

    # conditionally load to GPU
    if device == "cuda:0":
        print("loading to GPU...")
        model.cuda()
        print("done")

    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-1b3")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Tokenize inputs
    input_tokens = tokenizer(prompt, return_tensors="pt").to(device)

    # Run the model
    output = model(**input_tokens, labels=inputs_tokens["input_ids"])

    # Decode output tokens
    output_text = tokenizer.batch_decode(output, skip_special_tokens = True)[0]

    result = {"output": output_text}

    # Return the results as a dictionary
    return result
