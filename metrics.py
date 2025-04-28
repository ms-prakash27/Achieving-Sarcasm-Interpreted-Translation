import evaluate
import nltk
import numpy as np
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk import ngrams

def bleu_score(predictions, references):
    bleu = evaluate.load('bleu')
    return bleu.compute(predictions=predictions, references=references)['bleu']

def rouge_scores(predictions, references):
    rouge = evaluate.load('rouge')
    return rouge.compute(predictions=predictions, references=references)

def pinc_score(predictions, sources):
    """
    Calculate how deviant predictions are from the sources
    
    Based on the PINC score defined in https://www.cs.utexas.edu/users/ml/papers/chen.acl11.pdf
    """
    assert len(predictions) == len(sources), "predictions and sources must be of same length"
    
    N = 4 # Maximum n-grams
    
    pinc_scores = [0]
    
    for pred, source in zip(predictions, sources):
        pinc_ = 0
        for n in range(1, N + 1):
            pred_grams = set(' '.join(gram) for gram in ngrams(word_tokenize(pred), n))
            source_grams = set(' '.join(gram) for gram in ngrams(word_tokenize(source), n))
            
            if len(pred_grams) == 0:
                break # Zerodivision

            pinc_ += (1 - len(pred_grams & source_grams) / len(pred_grams)) / N
        pinc_scores.append(pinc_)

    return np.mean(pinc_scores)
