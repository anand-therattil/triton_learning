from pytriton.client import ModelClient
import numpy as np

# Connect to the model
with ModelClient("localhost", "qwen3_model") as client:
    prompt = "Give me a short introduction to large language model."
    
    # Prepare inputs with batch dimension
    text_input = np.array([[prompt.encode('utf-8')]], dtype=object)
    enable_thinking = np.array([[True]], dtype=bool)
    
    # Send request
    result_dict = client.infer_batch(
        text_input=text_input,
        enable_thinking=enable_thinking
    )
    print(result_dict)
    # PyTriton's ModelClient handles decoding automatically
    # thinking = result_dict["thinking_content"][0][0].decode('utf-8')
    # content = result_dict["content"][0][0].decode('utf-8')
    
    # print("="*50)
    # print("THINKING CONTENT:")
    # print("="*50)
    # print(thinking)
    # print("\n" + "="*50)
    # print("RESPONSE CONTENT:")
    # print("="*50)
    # print(content)
