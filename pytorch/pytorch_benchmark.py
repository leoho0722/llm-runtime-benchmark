import argparse
import time
from typing import Any, Dict, List, Tuple, Union

import torch
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    PreTrainedTokenizer, PreTrainedTokenizerFast,
    TextStreamer
)

# Load Model and Tokenizer
start_model_load_time: float = None
end_model_load_time: float = None
total_model_load_time: float = None

# Process Prompt
start_prompt_process_time: float = None
end_prompt_process_time: float = None
total_prompt_process_time: float = None
prompt_tokens: int = None
prompt_tps: float = None

# Model Inference
start_inference_time: float = None
end_inference_time: float = None
total_inference_time: float = None
time_to_first_token: float = None
generation_tokens: int = None
generation_tps: float = None


def load_model(
    model_id: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
    trust_remote_code: bool,
) -> Tuple[AutoModelForCausalLM, Union[PreTrainedTokenizer, PreTrainedTokenizerFast], float]:
    """Load Model and Tokenizer

    Args:
        model_id (`str`): Path to the model
        load_in_4bit (`bool`): Load the model in 4-bit
        load_in_8bit (`bool`): Load the model in 8-bit
        trust_remote_code (`bool`): Trust remote code

    Returns:
        model (`AutoModelForCausalLM`): Model
        tokenizer (`Union[PreTrainedTokenizer, PreTrainedTokenizerFast]`): Tokenizer
        total_model_load_time (`float`): Total time taken to load the model
    """

    if load_in_4bit and load_in_8bit:
        raise ValueError("Cannot load in both 4-bit and 8-bit")

    q_config: BitsAndBytesConfig = None

    if load_in_4bit or load_in_8bit:
        q_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit
        )

    start_model_load_time = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=q_config,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        cache_dir=HF_HUB_CACHE
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_HUB_CACHE)

    end_model_load_time = time.perf_counter()

    total_model_load_time = end_model_load_time - start_model_load_time

    return model, tokenizer, total_model_load_time


def process_prompt(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str):
    """Process the Prompt

    Args:
        model (`AutoModelForCausalLM`): Model
        tokenizer (`AutoTokenizer`): Tokenizer
        prompt (`str`): Prompt for the LLM model

    Returns:
        input_ids (`torch.Tensor`): Input IDs
        prompt_tokens (`int`): Prompt tokens length
        total_prompt_process_time (`float`): Total time taken to process the prompt
        prompt_tps (`float`): Tokens per second for the prompt processing
    """

    start_prompt_process_time = time.perf_counter()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = len(input_ids[0])

    end_prompt_process_time = time.perf_counter()
    total_prompt_process_time = end_prompt_process_time - start_prompt_process_time
    prompt_tps = round(prompt_tokens / total_prompt_process_time, 2)

    return input_ids, prompt_tokens, total_prompt_process_time, prompt_tps


def model_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    input_ids: torch.Tensor,
    max_new_tokens: int
) -> Dict[str, Any]:
    """Model Inference

    Args:
        model (`AutoModelForCausalLM`): Model
        tokenizer (`AutoTokenizer`): Tokenizer
        prompt (`str`): Prompt for the LLM model
        input_ids (`torch.Tensor`): Input IDs
        max_new_tokens (`int`): Maximum number of new tokens to generate

    Returns:
        output (`str`): Generated output
        generation_tokens (`int`): Number of tokens generated
        time_to_first_token (`float`): Time taken to generate the first token
        total_inference_time (`float`): Total time taken for inference
        generation_tps (`float`): Tokens per second for generation
    """

    start_inference_time = time.perf_counter()

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        top_k=10,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id
    )

    streamer = TextStreamer(tokenizer, skip_prompt=True)

    output = model.generate(
        input_ids, generation_config=generation_config, streamer=streamer)
    generation_tokens = len(output[0]) - len(input_ids[0])

    end_inference_time = time.perf_counter()

    total_inference_time = round(end_inference_time - start_inference_time, 2)

    if max_new_tokens == 1:
        time_to_first_token = total_inference_time

    generation_tps = round(generation_tokens / total_inference_time, 2)

    llm_output = tokenizer.decode(output[0], skip_special_tokens=True)
    llm_output = llm_output.replace(prompt, "")

    if max_new_tokens == 1:
        return {
            "output": llm_output,
            "generation_tokens": generation_tokens,
            "time_to_first_token": time_to_first_token,
            "generation_tps": generation_tps,
        }
    else:
        return {
            "output": llm_output,
            "generation_tokens": generation_tokens,
            "total_inference_time": total_inference_time,
            "generation_tps": generation_tps
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Benchmark using PyTorch and Hugging Face Transformers"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Path or Hugging Face Model Repository ID"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt for the LLM Model"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the model in 4-bit"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load the model in 8-bit"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        required=True,
        help="Maximum number of new tokens to generate"
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    model_id: str = args.model
    prompt: str = args.prompt
    load_in_4bit: bool = args.load_in_4bit
    load_in_8bit: bool = args.load_in_8bit
    trust_remote_code: bool = args.trust_remote_code
    max_tokens: int = args.max_tokens

    print("========== Benchmark Config ==========\n")
    print(f"Model: {model_id}")
    print(f"Prompt: {prompt}")
    print(f"Load in 4-bit: {load_in_4bit}")
    print(f"Load in 8-bit: {load_in_8bit}")
    print(f"Trust Remote Code: {trust_remote_code}")
    print(f"Max Tokens: {max_tokens}")
    print()

    # Load Model and Tokenizer
    model, tokenizer, total_model_load_time = load_model(
        model_id=model_id,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        trust_remote_code=trust_remote_code,
    )

    # Process Prompt
    input_ids, prompt_tokens, total_prompt_process_time, prompt_tps = process_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
    )

    # Model Inference
    max_new_tokens: List[int] = [1, max_tokens]
    benchmark_results: Dict[str, Any] = {}
    for max_tokens in max_new_tokens:
        result = model_inference(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            input_ids=input_ids,
            max_new_tokens=max_tokens,
        )
        if max_tokens == 1:
            time_to_first_token = result["time_to_first_token"]

        benchmark_results = result

    # Show Benchmark Results
    total_inference_time = benchmark_results["total_inference_time"]
    generation_tokens = benchmark_results["generation_tokens"]
    generation_tps = benchmark_results["generation_tps"]
    llm_output = benchmark_results["output"]

    print("\n\n========== Benchmark Results ==========\n")
    print(f"Total Model Load Time: {total_model_load_time :.2f} s")
    print(f"Prompt Tokens: {prompt_tokens} tokens")
    print(f"Total Prompt Process Time: {total_prompt_process_time :.2f} s")
    print(f"Prompt TPS: {prompt_tps} tokens/s")
    print(f"Total Inference Time: {total_inference_time :.2f} s")
    print(f"Time to First Token: {time_to_first_token :.2f} s")
    print(f"Generation Tokens: {generation_tokens} tokens")
    print(f"Generation TPS: {generation_tps} tokens/s")
    print(f"LLM Output: {llm_output}")
