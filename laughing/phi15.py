import torch
import transformers
from peft import LoraConfig, get_peft_model

model_name_or_path = 'microsoft/phi-1_5'

def load_model_and_tokenizer(lora_rank:int=None, model_max_length:int=2048):
    """Loads the base phi-1.5 model and its tokenizer.
    
    If `lora_rank` is given, this will return a Lora model.
    """
    print('loading model...')
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        trust_remote_code=True
    ).cuda()

    # Use Lora so we can train on smaller GPU    
    if lora_rank is not None:
        print(f'creating a peft model with lora (rank={lora_rank})...')
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=1,
            target_modules=["Wqkv"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        peft_model = get_peft_model(model, lora_config).cuda()
        peft_model.print_trainable_parameters()
    else:
        peft_model = None

    print('loading tokenizer...')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=model_max_length,
        padding_side="left",
        use_fast=False,
        torch_dtype="auto"
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, peft_model, tokenizer

def extract_program(response: str):
    """
    Extract the solution from phi1 model's response, as it often
    generates some random function after the required solution was generated.

    This could be improved further with more time.
    """
    # discard the original prompt as it is included in the response
    #response = response[len(prompt):]

    # get the result until the second def
    def1_pos = response.index('def ')
    try:
        def2_pos = response.index('def ', def1_pos+4)
    except ValueError as ex:
        def2_pos = len(response)
    return response[:def2_pos]

@torch.inference_mode()
def gen_n_responses(model, tokenizer, prompt, max_new_tokens=512, n_responses=10):
    """Generate multiple responses for the same prompt.
    
    This method uses a mix of greedy and sampling. 
    The greedy approach seems to generate a cohesive response, while beam-search
    is not supported by the model. So, generate 1 sequence using greedy
    and the rest using sampling.
    """
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to('cuda')
    seqs = []
    # greedy generation
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    text = tokenizer.batch_decode(outputs)[0]
    seqs.append(parse_response_phi15(text))
    # sampling generation
    if num_sequences >= 1:
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True,
                                do_sample=True, top_k=3, num_return_sequences=n_sequences-1)
        seqs.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))

    return seqs

@torch.inference_mode()
def gen_n_prompts(model, tokenizer, prompts, max_new_tokens=512, n_responses=10):
    """Generate a batch of prompts (one response for each prompt)."""
    inputs = tokenizer(prompts, return_tensors="pt", return_attention_mask=False,
                       padding=True, truncation=True).to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             eos_token_id=tokenizer.eos_token_id)
    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return texts

def make_prompts_for_code_gen(tasks):
    """Make prompts using an instruction only for generating code/program.

    This method just prompts the model using the instruction directly without specifying
    any additional context.

    Tasks should be data that come from code_alpaca_20k dataset.
    """
    prompt = 'Generate a Python program for the following instruction.\n'      
    return [prompt + task['instruction'] +'\nPython code: ' for task in tasks]

def make_prompts_for_tests_gen(tasks):
    """Make prompts using an instruction and its code/program to generate unit test cases.

    Tasks should be data that come from code_alpaca_20k dataset.
    """
    def make_prompt(task):
        return f"""Generate unit tests for the below problem and its solution in Python.
        Write each unit test as a seperate Python function with meaningful name that starts with 'test_'.

        Problem:
        {task['instruction']}

        Solution:
        {task['output']}
        """
    
    return [make_prompt(task) for task in tasks]

