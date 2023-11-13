# Import necessary libraries
import torch
import numpy as np
import evaluate
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Setting the DEVICE to cuda
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizers and models
tokenizer_toxicity = RobertaTokenizer.from_pretrained("SkolkovoInstitute/roberta_toxicity_classifier")
model_toxicity = RobertaForSequenceClassification.from_pretrained("SkolkovoInstitute/roberta_toxicity_classifier").to(DEVICE)
tokenizer_acceptability = AutoTokenizer.from_pretrained("iproskurina/tda-bert-en-cola")
model_acceptability = AutoModelForSequenceClassification.from_pretrained("iproskurina/tda-bert-en-cola").to(DEVICE)

# Initialize model variables
model_bleurt = None
model_bertscore = None
model_sacrebleu = None

def calc_sacrebleu(refs, preds):
    """
    Calculates the SacreBLEU score.

    Args:
        refs (list): List of reference sentences
        preds (list): List of predicted sentences
    
    Returns:
        results (float): SacreBLEU score
    """
    global model_sacrebleu

    if model_sacrebleu is None:
        model_sacrebleu = evaluate.load("sacrebleu")

    results = model_sacrebleu.compute(predictions=preds, references=refs)["score"]
    results = results/100

    return results

def calc_bert_score(
    refs, preds, model_type="microsoft/deberta-large-mnli", output_mean=True
    ):
    """
    Calculates BERT score per line. Note: https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit#gid=0 lists the best performing models
    Args:
        refs (list): List of reference sentences.
        y_pred (list): List of predicted sentences.
        model_type (str): Type of BERT model to use.
        output_mean (bool): Whether to output the mean of the scores.

    Returns:
        list of precision, recall, f1 scores.

    """
    global model_bertscore

    if model_bertscore is None:
        model_bertscore = evaluate.load("bertscore")
        
    results = model_bertscore.compute(predictions=preds, references=refs, model_type=model_type)
    precision = np.array(results["precision"])
    recall = np.array(results["recall"])
    f1 = np.array(results["f1"])
    
    if output_mean:
        precision = precision.mean()
        recall = recall.mean()
        f1 = f1.mean()

    return precision, recall, f1

def calc_bleurt(refs, preds, checkpoint="BLEURT-20_D12", output_mean = True):
    """
    Calculates BLEURT score per line.

    Args:
        refs (list): List of reference sentences.
        preds (list): List of predicted sentences.
        output_type (str): Type of output to return. Either 'numpy' or 'list'.

    Returns:
        list/array of BLEURT scores.
    """
    global model_bleurt

    if model_bleurt is None:
        model_bleurt = evaluate.load("bleurt", module_type="metric", checkpoint=checkpoint)

    results = np.array(model_bleurt.compute(predictions=preds, references=refs)["scores"])

    if output_mean:
        results = results.mean()

    return results

def calc_tox_acceptability(
    data,
    tokenizer,
    model,
    output_score=True,
    output_mean=True):
    """
    Calculates toxicity and acceptability scores for a given dataset.

    Args:
        data = list of strings to be evaluated
        tokenizer = tokenizer for the model
        model = model to be used for evaluation
        output_score = whether to output the score or the label
        output_mean = whether to output the mean of the scores or the scores for each sentence
    
    Returns:
        array of toxicity and acceptability scores.
    """  
    inputs = tokenizer(data, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs)["logits"]
        if output_score:
            result = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        else:
            result = logits.argmax(1).data
        result = result.cpu().numpy()

    if output_mean:
        result = result.mean()
        
    return result

def evaluate_metrics(
    refs,
    preds,
    tokenizer_toxicity=tokenizer_toxicity,
    model_toxicity=model_toxicity,
    tokenizer_acceptability=tokenizer_acceptability,
    model_acceptability=model_acceptability,
    to_neutral=True,
    weights={
        "BLEU": 0.2,
        "STA": 0.4,
        "Acceptability": 0.2,
        "BERT_Score": 0.2
    },
    include_bleurt=False
):
    """
    Calculates and returns a dictionary of evaluation metrics

    Args:
        refs (list): list of strings (reference)
        preds (list): list of strings (predictions)
        tokenizer_toxicity (tokenizer): tokenizer for toxicity model
        model_toxicity (model): toxicity model
        tokenizer_acceptability (tokenizer): tokenizer for acceptability model
        model_acceptability (model): acceptability model
        to_neutral (bool): whether the goal is to transfer to neutral (True) or to toxic (False)
        weights (dict): dictionary of weights for each metric
        include_bleurt (bool): whether to include BLEURT score in the output

    Returns:
        results (dict): dictionary of evaluation metrics
    """
    # Calculate BLEU score
    bleu = calc_sacrebleu(refs, preds)

    # Calculate toxicity classification
    tox_pred = calc_tox_acceptability(preds, tokenizer_toxicity, model_toxicity, output_score=False, output_mean=False)

    # Calculate style transfer accuracy as proportion of sentences that were correctly classified (as non-toxic / toxic)
    if to_neutral:
        sta_correct_label = 0
    else:
        sta_correct_label = 1

    sta_pred = (tox_pred == sta_correct_label).sum() / len(tox_pred)

    # Calculate acceptability scores
    acc_pred = calc_tox_acceptability(preds, tokenizer_acceptability, model_acceptability)

    # Calculate similarity score
    bert_score_f1 = calc_bert_score(refs, preds, model_type="distilbert-base-uncased")[2]

    # Calculate BLEURT score if include_bleurt is True
    bleurt = None
    if include_bleurt:
        bleurt = calc_bleurt(refs, preds)

    # Calculate composite score
    composite_score = weights["BLEU"] * bleu + weights["STA"] * sta_pred + weights["Acceptability"] * acc_pred + weights["BERT_Score"] * bert_score_f1

    # Return a dictionary of metrics
    results = {
        "BLEU": bleu,
        "STA": sta_pred,
        "FLU": acc_pred,
        "SEM": bert_score_f1,
        "Overall": composite_score,
    }
    if include_bleurt:
        results["BLEURT"] = bleurt
        
    return results