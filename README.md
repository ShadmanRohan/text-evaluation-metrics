# Text Evaluation Metrics (TEM)
Evaluation metrics for text data: BLEU is for machine translations and ROUGE is for text summaries.
This library provides implementations of the BLEU and ROUGE evaluation metrics, which are commonly used for evaluating the quality of machine translations and text summaries.

## Notes
* This implementation is for educational purpose
* Compute BLEU scores (bilingual evaluation understudy) for machine translations, which is a widely-used metric for evaluating the quality of machine-generated translations
* Compute ROUGE-N scores (recall-oriented understudy for gisting evaluation) for text summaries, which is a common metric for evaluating the quality of summarization models
* Tokenization and n-gram counting utilities. The package uses Python's split() method for tokenization.


## Installation
You can either clone the repository or download the tem.py file and place it in your project directory.

`git clone https://github.com/yourusername/text-evaluation-metrics.git`

## Usage
To use the library, simply import the tem module and call the bleu_score and rouge_score functions:


```python
import tem

candidate = "The quick brown dog jumped over the lazy fox."
references = [
    "The quick brown dog jumps over the lazy fox.",
    "The fast brown canine leaps over the lazy vulpine."
]

print("BLEU score:", tem.bleu_score(candidate, references))
print("ROUGE score:", tem.rouge_score(candidate, references, n=2))
```

