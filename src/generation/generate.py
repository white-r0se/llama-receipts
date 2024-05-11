import os
import sys

import torch
from peft import PeftModel

from transformers import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.trainer.train_qlora import HUMAN_PROMPT


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def generate(
    load_8bit: bool = False,
    base_model: str = "IlyaGusev/saiga_llama3_8b",
    lora_weights: str = "../models/checkpoint/",
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model'"

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if device == "cuda":
        for _ in range(5):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map={"": 0}
                )
            except:
                continue

        model = PeftModel.from_pretrained(model, lora_weights)
        model.eval()
        model = model.to(device)
        print("load model and tokenizer done!")


    else:
        for _ in range(5):
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model, device_map={"": device}, low_cpu_mem_usage=True
                )
            except:
                continue

        model = PeftModel.from_pretrained(model, lora_weights)
        model.eval()
        model = model.to(device)
        print("load model and tokenizer done!")

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        input=None,
        temperature=0.5,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=512,
        repetition_penalty=1.2,
        **kwargs,
    ) -> str:
        print("repetition_penalty: ", repetition_penalty)
        prompt = "{}{}{}".format(tokenizer.bos_token, input, tokenizer.eos_token)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        # Without streaming
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids,
                             max_new_tokens=max_new_tokens,
                             do_sample=True,
                             top_p=top_p,
                             temperature=temperature,
                             repetition_penalty=repetition_penalty,
                             eos_token_id=tokenizer.eos_token_id)
        outputs = outputs[:, input_ids.shape[-1]: ]
        results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        response = results[0].strip()
        return response

    for instruction in [
        HUMAN_PROMPT + "ЖЕВАТ.РЕЗИНКА ОРБИТ",
        HUMAN_PROMPT + "Смартфон\\\\nPDA Samsung SM-A225F/DSN 64GB Wh",
        HUMAN_PROMPT + "RED BULL 0,473Л Ж/Б",
        HUMAN_PROMPT + "КОНДИЦИОНЕР LG 12000 BTU",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
