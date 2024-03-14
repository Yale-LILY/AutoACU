import torch
import numpy as np
from transformers import BertTokenizer
import json
from tqdm import tqdm
from typing import List
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

class BertClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        top_vec = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        cls_vec = top_vec[:, 0, :]
        logits = self.linear(cls_vec).squeeze(-1)
        return logits

class A3CU():
    """
    Efficient and Interpretable Automatic Summarization Evaluation Metrics
    """
    def __init__(self, model_pt: str="Yale-LILY/a3cu", max_len: int=254, cpu: bool=False,
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """
        Args:
            model_pt: path to the HuggingFace model
            device: GPU device to use (default: 0)
            max_len: max length of the input
            cpu: use CPU instead of GPU
        """
        
        self.device = device
        self.model = BertClassifier.from_pretrained(model_pt).to(self.device)
        self.tok = BertTokenizer.from_pretrained(model_pt)
        self.model.eval()
        self.max_len = max_len

    def __prepare_batch(self, batch):
        # prepare batch
        def pad(X, pad_id, max_len=-1):
            if max_len < 0:
                max_len = max(x.size(0) for x in X)
            result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_id
            for (i, x) in enumerate(X):
                result[i, :x.size(0)] = x
            return result

        input_ids = pad([x["src_input_ids"] for x in batch], self.tok.pad_token_id).to(self.device)
        attention_mask = input_ids != self.tok.pad_token_id
        token_type_ids = pad([x["segs"] for x in batch], 0).to(self.device)
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        return result

    def __get_batch(self, references, candidates, batch_size=1):
        # yeild batchs
        for i in range(0, len(references), batch_size):
            batch = references[i:i+batch_size]
            cand_batch = candidates[i:i+batch_size]
            recall_batch, prec_batch = [], []
            for ref, cand in zip(batch, cand_batch):
                reference_summary_ids = self.tok(ref, add_special_tokens=False, truncation=True, max_length=self.max_len)
                candidate_summary_ids = self.tok(cand, add_special_tokens=False, truncation=True, max_length=self.max_len)
                src_input_ids = torch.tensor([self.tok.cls_token_id] + candidate_summary_ids["input_ids"] + [self.tok.sep_token_id] + reference_summary_ids["input_ids"] + [self.tok.sep_token_id])
                segs = torch.tensor([0] * (len(candidate_summary_ids["input_ids"]) + 2) + [1] * (len(reference_summary_ids["input_ids"]) + 1))
                recall_batch.append({
                    "src_input_ids": src_input_ids,
                    "segs": segs,
                })
                src_input_ids = torch.tensor([self.tok.cls_token_id] + reference_summary_ids["input_ids"] + [self.tok.sep_token_id] + candidate_summary_ids["input_ids"] + [self.tok.sep_token_id])
                segs = torch.tensor([0] * (len(reference_summary_ids["input_ids"]) + 2) + [1] * (len(candidate_summary_ids["input_ids"]) + 1))
                prec_batch.append({
                    "src_input_ids": src_input_ids,
                    "segs": segs,
                })
            recall_batch = self.__prepare_batch(recall_batch)
            prec_batch = self.__prepare_batch(prec_batch)
            yield {
                "recall_batch": recall_batch,
                "prec_batch": prec_batch,
            }


    def score(self, references: List[str], candidates: List[str], batch_size: int=1, output_path: str=None, verbose: bool=True):
        """
        evaluate a list of candidate summaries against a list of reference summaries
        Args:
            references: list of reference summaries
            candidates: list of candidate summaries
            batch_size: batch size
            output_path: path to save the scores
            verbose: show progress

        Returns:
            recall_scores: list of recall scores
            prec_scores: list of precision scores
            f1_scores: list of f1 scores
        """
        batchs = self.__get_batch(references, candidates, batch_size)
        num_batch = len(references) // batch_size
        if output_path is not None:
            f = open(output_path, "w")
        recall_scores, prec_scores, f1_scores = [], [], []
        # using tqdm to show progress
        with torch.no_grad():
            for batch in tqdm(batchs, total=num_batch, disable=not verbose):
                # recall
                input_ids = batch["recall_batch"]["input_ids"]
                attention_mask = batch["recall_batch"]["attention_mask"]
                token_type_ids = batch["recall_batch"]["token_type_ids"]
                logits = self.model(input_ids, attention_mask, token_type_ids)
                recall = logits.cpu().numpy()
                # precision
                input_ids = batch["prec_batch"]["input_ids"]
                attention_mask = batch["prec_batch"]["attention_mask"]
                token_type_ids = batch["prec_batch"]["token_type_ids"]
                logits = self.model(input_ids, attention_mask, token_type_ids)
                prec = logits.cpu().numpy()
                # f1
                f1 = 2 * prec * recall / (prec + recall + 1e-10)
                recall = recall.tolist()
                prec = prec.tolist()
                f1 = f1.tolist()
                recall_scores.extend(recall)
                prec_scores.extend(prec)
                f1_scores.extend(f1)
                if output_path is not None:
                    for r, p, f in zip(recall, prec, f1):
                        print(json.dumps({"recall": r, "precision": p, "f1": f}), file=f)
        if output_path is not None:
            f.close()
        if verbose:
            print("recall: %.4f, precision: %.4f, f1: %.4f" % (np.mean(recall_scores), np.mean(prec_scores), np.mean(f1_scores)))
        return recall_scores, prec_scores, f1_scores

    def score_example(self, reference: str, candidate: str):
        """
        evaluate a candidate summary against a reference summary

        Args:
            reference: reference summary
            candidate: candidate summary

        Returns:
            recall, precision, f1 score
        """
        recall_score, prec_score, f1_score = self.score([reference], [candidate], verbose=False)
        return recall_score[0], prec_score[0], f1_score[0]




        




    
        
