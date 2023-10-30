def eval_bleu(y_test, y_pred, smoothing=True):
    """
    Calculates overall BLEU score for the model

    Args:
        y_test (list): list of reference labels
        y_pred (list): list of predicted labels

    Returns:
        bleu_sim (float): BLEU score
    """
    bleu_sim = 0
    counter = 0
    if smoothing:
        for i in range(len(y_test)):
            if len(y_test[i]) > 3 and len(y_pred[i]) > 3:
                bleu_sim += sentence_bleu(
                    [y_test[i]],
                    y_pred[i],
                    smoothing_function=SmoothingFunction().method1,
                )
                counter += 1
    else:
        for i in range(len(y_test)):
            if len(y_test[i]) > 3 and len(y_pred[i]) > 3:
                bleu_sim += sentence_bleu([y_test[i]], y_pred[i])
                counter += 1
    return float(bleu_sim / counter)


print(f"BLEU score for baseline: {eval_bleu(y_test, y_pred_baseline)}")


def eval_sta(y):
    """
    Calculates the Style Accuracy score as the percentage of non-toxic outputs identified by the style classifier

    Args:
        y (list): list of toxicity labels

    Returns:
        y_toxlabel (list): list of toxicity labels
        sta (float): Style Accuracy score
    """
    y_toxlabel = classify_toxicity(
        y,
        tokenizer=tokenizer_toxicity,
        model=model_toxicity,
        batch_size=BATCH_SIZE,
        include_tqdm=True,
        output_type="numpy",
        flatten_output=True,
    )
    sta = 1 - (y_toxlabel.sum() / len(y_toxlabel))
    return y_toxlabel, sta


y_pred_baseline_toxlabel, sta_baseline = eval_sta(y_pred_baseline)
y_test_toxlabel, sta_test = eval_sta(y_test)

print(f"STA for baseline: {sta_baseline}")
print(f"STA for reference: {sta_test}")

# Need to rewrite this using huggingface evaluate method
def calc_bleurt(
    y_test, y_pred, bleurt_checkpoint="../models/BLEURT-20-D12", output_type="numpy", flatten_output=True
):
    """
    Calculates BLEURT score per line.

    Args:
        y_test (list): List of reference sentences.
        y_pred (list): List of predicted sentences.
        bleurt_checkpoint (str): Path to the BLEURT checkpoint directory.
        output_type (str): The type of output to return. 'Numpy', 'List'.
        flatten_output (bool): Whether to return a single float value if the input has only one sentence.

    Returns:
        np.array or list: Array or list of BLEURT scores per line.
    """

    # Create an instance of BLEURT scorer
    bleurt_scorer = score.LengthBatchingBleurtScorer(bleurt_checkpoint)

    # Convert y_test and y_pred to lists if they are not already
    if isinstance(y_test, (np.ndarray, str)):
        y_test = y_test.tolist() if isinstance(y_test, np.ndarray) else [y_test]
    if isinstance(y_pred, (np.ndarray, str)):
        y_pred = y_pred.tolist() if isinstance(y_pred, np.ndarray) else [y_pred]

    # Calculate BLEURT score per line
    scores = bleurt_scorer.score(references=y_test, candidates=y_pred)

    # Convert scores to numpy array or list    
    if output_type == "numpy":
        bleurt_scores = np.array(scores)
    elif output_type == "list":
        bleurt_scores = scores
    else:
        raise ValueError(f"output_type must be either 'numpy' or 'list', but got {output_type}.")

    # If flatten_output is True, return a single value if only one sentence was passed
    if flatten_output and len(bleurt_scores) == 1:
        bleurt_scores = bleurt_scores[0]

    return bleurt_scores

def calc_similarity(input_1, input_2, model="all-MiniLM-L6-v2", output_type="numpy", flatten_output=True):
    """
    Calculates cosine similarity between embedddings of two sentences

    Args:
        input_1 (str): first sentence or list of sentences
        input_2 (str): second sentence or list of sentences
        model (str): name of the model to be used for encoding
        ouput_type (str): type of output to return. 'numpy', 'list'
        flatten_output (bool): whether to return a single value if only one sentence was passed

    Returns:
        similarity (list/array): cosine similarity between the two sentences (or lists of sentences
    """
    # Cast inputs as lists
    if isinstance(input_1, str):
        input_1 = [input_1]
    if isinstance(input_2, str):
        input_2 = [input_2]

    # Encode inputs
    model = SentenceTransformer(model)
    with torch.no_grad():
        embeddings_1 = model.encode(input_1, convert_to_tensor=True, device=DEVICE, batch_size=BATCH_SIZE)
        torch.cuda.empty_cache()
        embeddings_2 = model.encode(input_2, convert_to_tensor=True, device=DEVICE, batch_size=BATCH_SIZE)
        torch.cuda.empty_cache()
        
    # Calculate cosine similarity
    similarity = torch.cosine_similarity(embeddings_1, embeddings_2).cpu()

    # Flatten output if only one sentence was passed
    if flatten_output and len(similarity) == 1:
        return similarity[0]
        
    # Cast to appropraite type
    if output_type == "numpy":
        similarity = similarity.numpy()
    elif output_type == 'list':
        similarity = similarity.tolist()

    return similarity


def calc_bleu(y_test, y_pred, weights=(0.25, 0.25, 0.25, 0.25), output_type="numpy", flatten_output=True):
    """
    Calculates BLEU score per line.

    Args:
        y_test (list): List of reference sentences.
        y_pred (list): List of predicted sentences.

    Returns:
        list/array of BLEU scores.
    """
    # Create an instance of SmoothingFunction
    smoothing_function = SmoothingFunction().method1

    # Calculate BLEU score per line
    if output_type == "numpy":
        bleu_scores = np.empty(len(y_test), dtype=float)
        for i, (reference, hypothesis) in enumerate(zip(y_test, y_pred)):
            bleu_scores[i] = sentence_bleu(
                references=[reference],
                hypothesis=hypothesis,
                weights=weights,
                smoothing_function=smoothing_function,
            )

    elif output_type == "list":
        bleu_scores = []
        for reference, hypothesis in zip(y_test, y_pred):
            bleu_scores.append(
                sentence_bleu(
                    references=[reference],
                    hypothesis=hypothesis,
                    weights=weights,
                    smoothing_function=smoothing_function,
                )
            )

    else:
        raise ValueError(f"output_type must be either 'numpy' or 'list', but got {output_type}.")

    # If flatten_output is True, return a single value if only one sentence was passed
    if flatten_output and len(bleu_scores) == 1:
        bleu_scores = bleu_scores[0]

    return bleu_scores

def evaluate_metrics(
    y_test,
    y_pred,
    tokenizer_toxicity=tokenizer_toxicity,
    model_toxicity=model_toxicity,
    tokenizer_acceptability=tokenizer_acceptability,
    model_acceptability=model_acceptability,
    weights={
        "BLEU": 0.2,
        "Toxicity": 0.4,
        "Acceptability": 0.2,
        "BERT_Score": 0.2
    },
    include_bleurt=False
):
    """
    Calculates and returns a dictionary of evaluation metrics

    Args:
        y_test (list): list of strings
        y_pred (list): list of strings
        tokenizer_toxicity (tokenizer): tokenizer for toxicity model
        model_toxicity (model): toxicity model
        tokenizer_acceptability (tokenizer): tokenizer for acceptability model
        model_acceptability (model): acceptability model
        weights (dict): dictionary of weights for each metric
        include_bleurt (bool): whether to include BLEURT score in the output

    Returns:
        results (dict): dictionary of evaluation metrics
    """

    # Calculate BLEU score
    bleu = calc_sacrebleu(y_test, y_pred)

    # Calculate toxicity scores
    tox_y_test = calc_tox_acceptability(y_test, tokenizer_toxicity, model_toxicity)
    tox_y_pred = calc_tox_acceptability(y_pred, tokenizer_toxicity, model_toxicity)

    # Calculate style transfer score as 1 - toxicity score
    style_transfer_y_test = 1 - tox_y_test
    style_transfer_y_pred = 1 - tox_y_pred
    style_transfer_diff = style_transfer_y_pred - style_transfer_y_test

    # Calculate acceptability scores
    acc_y_test = calc_tox_acceptability(y_test, tokenizer_acceptability, model_acceptability)
    acc_y_pred = calc_tox_acceptability(y_pred, tokenizer_acceptability, model_acceptability)

    # Calculate difference
    acc_diff = acc_y_pred - acc_y_test

    # Calculate similarity score
    bert_score_precision, bert_score_recall, bert_score_f1 = calc_bert_score(y_test, y_pred, model_type="distilbert-base-uncased")

    # Calculate BLEURT score if include_bleurt is True
    bleurt = None
    if include_bleurt:
        bleurt = calc_bleurt(y_test, y_pred)

    # Calculate composite score
    composite_score = weights["BLEU"] * bleu + weights["Toxicity"] * (1-tox_y_pred) + weights["Acceptability"] * acc_diff + weights["BERT_Score"] * bert_score_f1

    # Return a dictionary of metrics
    results = {
        "BLEU": bleu,
        "Toxicity_y_pred": tox_y_pred,
        "Toxicity_diff": tox_diff,
        "Acceptability_y_pred": acc_y_pred,
        "Acceptability_diff": acc_diff,
        "BERT_score_f1": bert_score_f1,
        "Overall": composite_score,
    }
    if include_bleurt:
        results["BLEURT"] = bleurt

    return results
