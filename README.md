# AutoACU - Interpretable and Efficient Automatic Summarization Evaluation

This github repository contains the source code of the AutoACU package for automatic summarization evaluation, proposed in our paper [Towards Interpretable and Efficient Automatic Reference-Based Summarization Evaluation](https://aclanthology.org/2023.emnlp-main.1018/), EMNLP 2023


AutoACU contains two types of automatic evaluation metrics:
- **A2CU**: a two-step automatic evaluation metric that first extracts atomic content units (ACUs) from one text sequence and then evaluates the extracted ACUs against another text sequence.
- **A3CU**: an accelerated version of A2CU that directly computes the similarity between two text sequences without extracting ACUs, but with the similar evaluation target.

## Installation
You can install AutoACU using pip:
```bash
pip install autoacu
```
or clone the repository and install it manually:
```bash
git clone https://github.com/Yale-LILY/AutoACU
cd AutoACU
pip install .
```
The necessary dependencies include PyTorch and HuggingFace's Transformers.
It should be compatible with any of the recent versions of PyTorch and Transformers.
However, to make sure that the dependencies are compatible,
you may run the following command:
```bash
pip install autoacu[stable]
```
You may also use the metrics directly without installing the package by importing the metric classes in `autoacu/a2cu.py` and `autoacu/a3cu.py`.

## Usage

The model checkpoints for A2CU and A3CU are available on the HuggingFace model hub.

### A2CU
A2CU needs to be initialized with two models, one for ACU generation and one for ACU matching.
The default models are the following:
- ACU generation: [Yale-LILY/a2cu-generator](https://huggingface.co/Yale-LILY/a2cu-generator), which is a [T0-3B](https://huggingface.co/bigscience/T0_3B) model finetuned on the [RoSE](https://github.com/Yale-LILY/ROSE) dataset.
- ACU matching: [Yale-LILY/a2cu-classifier](https://huggingface.co/Yale-LILY/a2cu-classifier), which is a [DeBERTa-XLarge](https://huggingface.co/microsoft/deberta-xlarge-mnli) model finetuned on the RoSE dataset.

Please note that to use A2CU, you may need to have a GPU with at least 16GB memory.

Below is an example of using A2CU to evaluate the similarity between two text sequences.
```python
from autoacu import A2CU
candidates = ["Real Madrid have fought off all the competition to win the 2023/24 UEFA Champions League after beating Borussia Dortmund 2-0 in the final at Wembley Stadium on 1 June."]
references = ["On June 1st, Real Madrid emerged victorious in the 2023/24 UEFA Champions League, defeating Borussia Dortmund 2-0 in the final at Wembley Stadium, overcoming all competitors to claim the title."]
a2cu = A2CU(device=0)  # the GPU device to use
recall_scores, prec_scores, f1_scores = a2cu.score(
    references=references,
    candidates=candidates,
    generation_batch_size=2, # the batch size for ACU generation
    matching_batch_size=16, # the batch size for ACU matching
    output_path=None, # the path to save the evaluation results
    recall_only=False, # whether to only compute the recall score
    acu_path=None, # the path to save the generated ACUs
    )
print(f"Recall: {recall_scores[0]:.4f}, Precision {prec_scores[0]:.4f}, F1: {f1_scores[0]:.4f}")
```

### A3CU
The default model checkpoint for A3CU is [Yale-LILY/a3cu](https://huggingface.co/Yale-LILY/a3cu), which is based on the [BERT-Large](https://huggingface.co/bert-large-cased) model.
Below is an example of using A3CU to evaluate the similarity between two text sequences.
```python
from autoacu import A3CU
candidates = ["Real Madrid have fought off all the competition to win the 2023/24 UEFA Champions League after beating Borussia Dortmund 2-0 in the final at Wembley Stadium on 1 June."]
references = ["On June 1st, Real Madrid emerged victorious in the 2023/24 UEFA Champions League, defeating Borussia Dortmund 2-0 in the final at Wembley Stadium, overcoming all competitors to claim the title."]
a3cu = A3CU(device=0)  # the GPU device to use
recall_scores, prec_scores, f1_scores = a3cu.score(
    references=references,
    candidates=candidates,
    batch_size=16, # the batch size for ACU generation
    output_path=None, # the path to save the evaluation results
    )
print(f"Recall: {recall_scores[0]:.4f}, Precision {prec_scores[0]:.4f}, F1: {f1_scores[0]:.4f}")
```

### Warning
A2CU and A3CU may not work properly on short text sequences (e.g., less than 10 words) due to the limitations of the ACU generation model, especially for non-reducible text sequences.
Please see https://github.com/Yale-LILY/AutoACU/issues/1 for details.
