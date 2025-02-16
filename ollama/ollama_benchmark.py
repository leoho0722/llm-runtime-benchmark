import argparse
import asyncio
from typing import Any, Dict, List, Optional

from ollama import AsyncClient, Options


class OllamaClient:

    _aclient: AsyncClient = None

    _time_to_first_token: float = None

    def __init__(self, host: str):
        self._aclient = AsyncClient(host=host)

    async def stream(
        self,
        model: str,
        prompt: str,
        num_ctx: int = 2048,
        num_predict: Optional[int] = None,
        temperature: float = 0.0
    ):
        async for chunk in await self._aclient.generate(
            model=model,
            prompt=prompt,
            stream=True,
            options=Options(
                num_ctx=num_ctx,
                num_predict=num_predict,
                temperature=temperature,
            ),
            keep_alive="0s"
        ):
            if bool(chunk["done"]):
                formated_response = self._format_response(chunk)
                self._print_result(formated_response)
            else:
                if num_predict == 1:
                    print(chunk["response"], end="")
                    print("\n")
                else:
                    print(chunk["response"], end="")

    def _format_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        formated_response: Dict[str, Any] = {}

        for key, value in response:
            if (key == "model"):
                model = value
                formated_response[key] = model
            elif (key == "total_duration"):
                total_duration = float(value)/(10**9)
                formated_response[key] = total_duration
            elif (key == "load_duration"):
                load_duration = float(value)/(10**9)
                formated_response[key] = load_duration
            elif (key == "prompt_eval_count"):
                prompt_eval_count = int(value)
                formated_response[key] = prompt_eval_count
            elif (key == "prompt_eval_duration"):
                prompt_eval_duration = float(value)/(10**9)
                formated_response[key] = prompt_eval_duration
            elif (key == "eval_count"):
                eval_count = int(value)
                formated_response[key] = eval_count
            elif (key == "eval_duration"):
                eval_duration = float(value)/(10**9)
                formated_response[key] = eval_duration

        return formated_response

    def _print_result(self, formated_response: dict):
        model = formated_response["model"]
        total_duration = formated_response["total_duration"]
        load_duration = formated_response["load_duration"]
        prompt_eval_count = formated_response["prompt_eval_count"]
        prompt_eval_duration = formated_response["prompt_eval_duration"]
        prompt_eval_rate = (prompt_eval_count / prompt_eval_duration)
        eval_count = formated_response["eval_count"]
        eval_duration = formated_response["eval_duration"]
        eval_rate = (eval_count / eval_duration)
        if eval_count == 1:
            self._time_to_first_token = eval_duration
            return

        print("\n")
        print(f"model:                {model}")
        print(f"total duration:       {total_duration :.2f} s")
        print(f"load duration:        {load_duration :.2f} s")
        print(f"prompt eval count:    {prompt_eval_count} tokens")
        print(f"prompt eval duration: {prompt_eval_duration :.2f} s")
        print(f"prompt eval rate:     {prompt_eval_rate :.2f} tokens/s")
        print(f"eval count:           {eval_count} tokens")
        print(f"eval duration:        {eval_duration :.2f} s")
        print(f"eval rate:            {eval_rate :.2f} tokens/s")
        print(f"time to first token:  {self._time_to_first_token :.2f} s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Benchmark using Ollama"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Model name"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt for the LLM Model"
    )
    parser.add_argument(
        "--num_ctx",
        type=int,
        default=2048,
        help="Number of Context Length"
    )
    parser.add_argument(
        "--num_predict",
        type=int,
        default=None,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--ollama_host",
        type=str,
        default="http://localhost:11434",
        help="Ollama Host"
    )

    return parser.parse_args()


async def main(args: argparse.Namespace):
    model: str = args.model
    prompt: str = args.prompt
    num_ctx: int = args.num_ctx
    num_predict: Optional[int] = args.num_predict
    ollama_host: str = args.ollama_host

    max_new_tokens: List[int] = [1, num_predict]

    client = OllamaClient(ollama_host)
    for max_tokens in max_new_tokens:
        await client.stream(model, prompt, num_ctx, max_tokens)

if __name__ == "__main__":
    args = parse_args()

    asyncio.run(main(args))
