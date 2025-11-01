from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Load model
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

@batch
def infer_fn(text_input, enable_thinking=None):
    """Inference function for Qwen model."""
    results_thinking = []
    results_content = []
    
    for prompt in text_input:
        prompt_str = prompt[0].decode('utf-8')
        
        # Prepare messages
        messages = [{"role": "user", "content": prompt_str}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Generate
        generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Parse thinking content
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        thinking_content = tokenizer.decode(
            output_ids[:index], skip_special_tokens=True
        ).strip("\n")
        content = tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip("\n")
        
        results_thinking.append(thinking_content.encode('utf-8'))
        results_content.append(content.encode('utf-8'))
    
    return {
        "thinking_content": np.array(results_thinking, dtype=object),
        "content": np.array(results_content, dtype=object)
    }

# Start Triton with PyTriton
with Triton() as triton:
    triton.bind(
        model_name="qwen3_model",
        infer_func=infer_fn,
        inputs=[
            Tensor(name="text_input", dtype=np.bytes_, shape=(1,)),
            Tensor(name="enable_thinking", dtype=np.bool_, shape=(1,), optional=True),
        ],
        outputs=[
            Tensor(name="thinking_content", dtype=np.bytes_, shape=(-1,)),
            Tensor(name="content", dtype=np.bytes_, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=8)
    )
    triton.serve()
