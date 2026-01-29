import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class GPT2Speaker:
    def __init__(self, model_name="gpt2", device=None):
        self.model_name = model_name

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def truncate_document(self, text, max_doc_tokens=384):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids = ids[:max_doc_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def build_prompt(self, document, token_budget=30):
        # one-sentence summarization
        return (
            "Summarize the following news article in one sentence (<= {k} tokens).\n\n"
            "{doc}\n\nSummary:"
        ).format(k=token_budget, doc=document)

    @torch.inference_mode()
    def generate_candidates_batch(self,
                                 prompts,
                                 n_candidates=2,
                                 max_new_tokens=40,
                                 mode="sample",
                                 seed=13,
                                 temperature=0.9,
                                 top_p=0.95,
                                 top_k=50,
                                 num_beams=4,
                                 diversity_penalty=0.3):
        """
        Returns: list of lists, len == batch_size; each inner list has n_candidates strings.
        """
        self.set_seed(seed)

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        input_len = enc["input_ids"].shape[1]  # padded length

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": n_candidates,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if mode == "sample":
            gen_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            })
        elif mode == "beam":
            # diverse beam search for multiple candidates
            nb = max(num_beams, n_candidates)
            ng = min(n_candidates, nb)
            while nb % ng != 0:
                nb += 1
            gen_kwargs.update({
                "do_sample": False,
                "num_beams": nb,
                "num_beam_groups": ng,
                "diversity_penalty": diversity_penalty,
                "early_stopping": True,
            })
        else:
            raise ValueError("mode must be 'sample' or 'beam'")

        sequences = self.model.generate(**enc, **gen_kwargs)

        # slice off prompt part
        completions = sequences[:, input_len:]
        texts = self.tokenizer.batch_decode(completions, skip_special_tokens=True)

        # group back into per-prompt lists
        batch_size = len(prompts)
        out = []
        idx = 0
        for _ in range(batch_size):
            out.append([texts[idx + j].strip() for j in range(n_candidates)])
            idx += n_candidates
        return out
