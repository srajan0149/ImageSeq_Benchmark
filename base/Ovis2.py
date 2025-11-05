import torch
from PIL import Image
from transformers import AutoModelForCausalLM

class Ovis2:
    def __init__(self, model_path: str, prompt:str):
        self.model_path = model_path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                multimodal_max_length=32768,
                trust_remote_code=True
        ).to(device)
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.prompt = '<image>\n'*5 + prompt


    def preprocess_images(self, paths:list[str]):
        images = [Image.open(path) for path in paths]
        return images


    def order_images(self, paths: list[str]):
        images = self.preprocess_images(paths)
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
                self.prompt,
                images,
                max_partition=12
                )
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)
        pixel_values = [pixel_values]

        # generate output
        with torch.inference_mode():
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=True,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True
            )
            output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
            output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
            return output
