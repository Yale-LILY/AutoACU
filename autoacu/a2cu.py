from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm
import json
import os
from transformers import DebertaForSequenceClassification, DebertaTokenizer
import tempfile
from typing import List
import numpy as np

class A2CU():
    """
    Automatic ACU Generation and Matching
    """
    def __init__(self, generation_pt: str="Yale-LILY/a2cu-generator",
                  matching_pt: str="Yale-LILY/a2cu-classifier", device: int=0, no_ref: bool=True):
        """
        Args:
            generation_pt: path to the HuggingFace model for generation
            matching_pt: path to the HuggingFace model for matching
            device: GPU device to use (default: 0)
            no_ref: whether to use reference summary
        """
        self.device = device
        self.generation_model = T5ForConditionalGeneration.from_pretrained(generation_pt)
        self.generation_model.eval()
        self.generation_tok = T5Tokenizer.from_pretrained("bigscience/T0_3B")
        self.matching_model = DebertaForSequenceClassification.from_pretrained(matching_pt)
        self.matching_model.eval()
        self.matching_tok = DebertaTokenizer.from_pretrained("microsoft/deberta-xlarge-mnli")
        self.no_ref = no_ref

    def acu_generation(self, ref_path: str, output_path: str, batch_size: int):
        """
        generate ACUs from reference summaries and save to output_path as jsonl

        Args:
            ref_path: path to the reference summary
            output_path: path to the output file
            batch_size: batch size

        Returns:
            list of ACUs
        """
        self.generation_model = self.generation_model.to(self.device)
        def tokenize(x):
            return self.generation_tok(x, add_special_tokens=True, truncation=True, max_length=256)["input_ids"]

        def pad(X, max_len=-1):
            if max_len < 0:
                max_len = max(x.size(0) for x in X)
            result = torch.ones(len(X), max_len, dtype=X[0].dtype) * self.generation_tok.pad_token_id
            for (i, x) in enumerate(X):
                result[i, :x.size(0)] = x
            return result
        
        def data_loader():
            with open(ref_path) as f:
                batch = []
                for line in f:
                    batch.append(line.strip())
                    if len(batch) == batch_size:
                        references = pad([torch.tensor(tokenize(x)) for x in batch]).to(self.device)
                        yield {
                            "src_input_ids": references,
                        }
                        batch = []
                if len(batch) > 0:
                    references = pad([torch.tensor(tokenize(x)) for x in batch]).to(self.device)
                    yield {
                        "src_input_ids": references,
                    }

        dataloader = data_loader()
        with open(ref_path) as f:
            num_lines = sum(1 for _ in f)
        num_batches = num_lines // batch_size
        acus = []

        with torch.no_grad():
            with open(output_path, "w") as f:
                for batch in tqdm(dataloader, total=num_batches):
                    text_id = batch["src_input_ids"]
                    input_mask = text_id != self.generation_tok.pad_token_id
                    summaries = self.generation_model.generate(
                        input_ids=text_id,
                        attention_mask=input_mask,
                        max_length=512,
                        min_length=10,
                        num_beams=4,
                        length_penalty=1.0,
                        early_stopping=True,
                    )
                    dec = [self.generation_tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                    for (i, x) in enumerate(dec):
                        print(json.dumps({
                            "acus": x.split("|||"),
                        }), file=f, flush=True)
                        acus.append(x.split("|||"))
        self.generation_model = self.generation_model.cpu()
        return acus


    def acu_matching(self, ref_path: str, cand_path: str, acu_path: str, output_path: str, output_dir: str, batch_size: int):
        """
        match ACUs with the candidate summaries and save to output_path as jsonl containing ACUs scores

        Args:
            ref_path: path to the reference summary
            cand_path: path to the candidate summary
            acu_path: path to the ACUs
            output_path: path to the output file
            output_dir: path to the output directory for intermediate files
            batch_size: batch size

        Returns:
            list of ACUs labels
        """
        self.matching_model = self.matching_model.to(self.device)
        gpuid = self.device
        no_ref = self.no_ref
        _acu_path = os.path.join(output_dir, "acu.tmp")
        _ref_path = os.path.join(output_dir, "ref.tmp")
        _cand_path = os.path.join(output_dir, "cand.tmp")
        with open(_acu_path, "w") as f, open(acu_path) as f_acu, open(ref_path) as f_ref, open(cand_path) as f_cand, open(_ref_path, "w") as f_ref_, open(_cand_path, "w") as f_cand_:
            for line in f_acu:
                ref_line = f_ref.readline().strip()
                cand_line = f_cand.readline().strip()
                acus = json.loads(line.strip())["acus"]
                for acu in acus:
                    print(acu.strip(), file=f)
                    print(ref_line, file=f_ref_)
                    print(cand_line, file=f_cand_)

        tok = self.matching_tok
        with open(_acu_path) as f:
            num_lines = sum(1 for _ in f)

        def pad(X, pad_id, max_len=-1):
            if max_len < 0:
                max_len = max(x.size(0) for x in X)
            result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_id
            for (i, x) in enumerate(X):
                result[i, :x.size(0)] = x
            return result

        def data_loader():
            with open(_ref_path) as f_ref, open(_cand_path) as f_cand, open(_acu_path) as f_acu:
                for i in range(0, num_lines, batch_size):
                    input_ids, token_type_ids = [], []
                    for j in range(batch_size):
                        ref = f_ref.readline().strip()
                        cand = f_cand.readline().strip()
                        acu = f_acu.readline().strip()
                        cand_ids = tok(cand, add_special_tokens=False, truncation=True, max_length=200)
                        ref_ids = tok(ref, add_special_tokens=False, truncation=True, max_length=200)
                        acu_ids = tok(acu, add_special_tokens=False, truncation=True, max_length=100)
                        if no_ref:
                            src_input_ids = [tok.cls_token_id] + cand_ids["input_ids"] + [tok.sep_token_id] + acu_ids["input_ids"] + [tok.sep_token_id]
                            segs = [0] * (len(cand_ids["input_ids"]) + 2) + [1] * (len(acu_ids["input_ids"]) + 1)
                        else:
                            src_input_ids = [tok.cls_token_id] + ref_ids["input_ids"] + [tok.sep_token_id] + cand_ids["input_ids"] + [tok.sep_token_id] + acu_ids["input_ids"] + [tok.sep_token_id]
                            segs = [0] * (len(ref_ids["input_ids"]) + 2) + [1] * (len(cand_ids["input_ids"]) + 1) + [0] * (len(acu_ids["input_ids"]) + 1)
                        src_input_ids = torch.tensor(src_input_ids)
                        segs = torch.tensor(segs)
                        input_ids.append(src_input_ids)
                        token_type_ids.append(segs)
                    input_ids = pad(input_ids, tok.pad_token_id).to(f"cuda:{gpuid}")
                    token_type_ids = pad(token_type_ids, 0).to(f"cuda:{gpuid}")
                    attenion_mask = (input_ids != tok.pad_token_id)
                    yield {
                        "input_ids": input_ids,
                        "token_type_ids": token_type_ids,
                        "attention_mask": attenion_mask,
                    }

        dataloader = data_loader()
        model = self.matching_model
        _output_path = os.path.join(output_dir, "acu_label.tmp")
        with open(_output_path, "w") as f:
            with torch.no_grad():
                for batch in tqdm(dataloader, total=num_lines // batch_size):
                    logits = model(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"], return_dict=True).logits
                    probs = torch.softmax(logits, dim=1)
                    entailment_probs = probs[:, 2] > 0.5
                    entailment_probs = entailment_probs.long().cpu().numpy().tolist()
                    for p in entailment_probs:
                        print(p, file=f)
        all_labels = []
        with open(acu_path) as f_acu, open(_output_path) as f_label, open(output_path, "w") as f:
            for acu_line in f_acu:
                acus = json.loads(acu_line.strip())["acus"]
                labels = [float(f_label.readline().strip()) for _ in range(len(acus))]
                score = sum(labels) / len(labels)
                print(score, file=f)
                all_labels.append(labels)
        self.matching_model = self.matching_model.cpu()
        return all_labels


    def __recall_score(self, references: List[str], candidates: List[str], generation_batch_size: int=1,
                   matching_batch_size: int=1, output_path: str=None, verbose: bool=True, acu_path: str=None):
        """
        Compute the recall score of the candidates given the references.

        Args:
            references: a list of reference strings
            candidates: a list of candidate strings
            generation_batch_size: the batch size for ACU generation
            matching_batch_size: the batch size for ACU matching
            output_path: the path to save the output scores
            verbose: whether to print the scores
            acu_path: the path to save the generated ACUs

        Returns:
            the recall scores
        """
        # create tmp_path dir using tempdir
        with tempfile.TemporaryDirectory() as tmp_path:
            # generate acus
            ref_path = os.path.join(tmp_path, "reference.txt")
            with open(ref_path, "w") as f:
                for x in references:
                    print(x.strip(), file=f)
            if acu_path is None:
                acu_gen_path = os.path.join(tmp_path, "acus.jsonl")
            else:
                acu_gen_path = acu_path
            self.acu_generation(ref_path, acu_gen_path, generation_batch_size)
            acus = []
            with open(acu_gen_path) as f:
                acus = [json.loads(line.strip())["acus"] for line in f]
            # match acus
            # generate input for match
            cand_path = os.path.join(tmp_path, "candidate.txt")
            acu_path = os.path.join(tmp_path, "acu.tmp.jsonl")
            with open(ref_path, "w") as f_ref, open(cand_path, "w") as f_cand, open(acu_path, "w") as f_acu:
                for i in range(len(references)):
                    print(references[i].strip(), file=f_ref)
                    print(candidates[i].strip(), file=f_cand)
                    print(json.dumps({"acus": acus[i]}), file=f_acu)
            # match
            acu_match_path = os.path.join(tmp_path, "scores.txt")
            self.acu_matching(ref_path, cand_path, acu_path, acu_match_path, tmp_path, matching_batch_size)
            # read scores
            scores = []
            with open(acu_match_path) as f:
                scores = [float(line.strip()) for line in f]
        if output_path is not None:
            with open(output_path, "w") as f:
                for score in scores:
                    print(json.dumps({"recall_score": score}), file=f)
        return scores

    def score(self, references: List[str], candidates: List[str], generation_batch_size: int=1, matching_batch_size: int=1,
               output_path: str=None, recall_only: bool=True, verbose: bool=True, acu_path: str=None):
        """
        Compute the A2CU score of the candidates given the references and save the scores to the output_path as jsonl if specified.

        Args:
            references: a list of reference summaries
            candidates: a list of candidate summaries
            generation_batch_size: the batch size for ACU generation
            matching_batch_size: the batch size for ACU matching
            output_path: the path to save the output scores
            recall_only: whether to only compute the recall score
            verbose: whether to print the scores
            acu_path: the path to save the generated ACUs (only used when recall_only is True)

        Returns:
            the recall scores, precision scores, and f1 scores as lists, or the recall scores if recall_only is True
        """
        if recall_only:
            scores = self.__recall_score(references, candidates, generation_batch_size, matching_batch_size, output_path, verbose, acu_path)
            if verbose:
                print(f"Recall score: {np.mean(scores).item():.4f}")
            return scores
        recall_scores = self.__recall_score(references, candidates, generation_batch_size, matching_batch_size, None, verbose)
        prec_scores = self.__recall_score(candidates, references, generation_batch_size, matching_batch_size, None, verbose)
        f1_scores = [2 * (recall_scores[i] * prec_scores[i]) / (recall_scores[i] + prec_scores[i] + 1e-10) for i in range(len(recall_scores))]
        if output_path is not None:
            with open(output_path, "w") as f:
                for recall, prec, f1 in zip(recall_scores, prec_scores, f1_scores):
                    print(json.dumps({"recall_score": recall, "precision_score": prec, "f1_score": f1}), file=f)
        if verbose:
            print(f"Recall score: {np.mean(recall_scores).item():.4f}")
            print(f"Precision score: {np.mean(prec_scores).item():.4f}")
            print(f"F1 score: {np.mean(f1_scores).item():.4f}")
        return recall_scores, prec_scores, f1_scores
    
    def score_example(self, reference: str, candidate: str):
        """
        Compute the A2CU score of the candidate given the reference.

        Args:
            reference: a reference summary
            candidate: a candidate summary

        Returns:
            the recall score, precision score, and f1 score
        """
        recall, prec, f1 = self.score([reference], [candidate], 1, 1, None, False, False)
        return recall[0], prec[0], f1[0]




    
