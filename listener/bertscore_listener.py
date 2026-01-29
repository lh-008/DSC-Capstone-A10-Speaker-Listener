import torch
from bert_score import score as bert_score


class BERTScoreListener:
    def __init__(self,
                 model_type="roberta-base",
                 lang="en",
                 device=None,
                 batch_size=16,
                 rescale_with_baseline=True):
        self.model_type = model_type
        self.lang = lang
        self.batch_size = batch_size
        self.rescale_with_baseline = rescale_with_baseline

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    @torch.inference_mode()
    def score_f1_batch(self, candidates, references):
        if len(candidates) != len(references):
            raise ValueError("candidates and references must match length")

        P, R, F1 = bert_score(
            candidates,
            references,
            model_type=self.model_type,
            lang=self.lang,
            device=self.device,
            batch_size=self.batch_size,
            rescale_with_baseline=self.rescale_with_baseline
        )
        return [float(x) for x in F1.detach().cpu().tolist()]
