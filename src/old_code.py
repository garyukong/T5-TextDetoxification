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

class Seq2SeqTrainerCustomLoss(Seq2SeqTrainer):
    def __init__(self, tokenizer_toxicity, model_toxicity, *args, **kwargs):
        super(Seq2SeqTrainerCustomLoss, self).__init__(*args, **kwargs)
        self.tokenizer_toxicity = tokenizer_toxicity
        self.model_toxicity = model_toxicity.to(DEVICE)

    def prediction_step_custom(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else False
        )

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        if has_labels:
            with self.compute_loss_context_manager():
                outputs = model(**inputs)
            if self.label_smoother is not None:
                loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
            else:
                loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
        else:
            loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        return loss, generated_tokens, labels

    def classifier_guided_loss(self, generated_text, target_label):
        # Tokenize the generated text
        inputs = self.tokenizer_toxicity(generated_text, return_tensors="pt", padding=True)
        
        # Move inputs to the same device as the model
        inputs = {key: tensor.to(DEVICE) for key, tensor in inputs.items()}
        
        # Get the batch size
        batch_size = inputs["input_ids"].shape[0]

        # Ensure target label is a tensor and on the correct device and shape
        target_label = torch.tensor([target_label]*batch_size, dtype=torch.long).to(DEVICE)

        # Get the classifier's output logits
        logits = self.model_toxicity(**inputs)["logits"]
        
        # Calculate the cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss().to(DEVICE)
        loss = loss_fct(logits, target_label)
        
        return loss
    
    def compute_loss(self, model, inputs):
        """
        Compute custom loss for the model.

        Args:
            model (torch.nn.Module): The model training or evaluating.
            inputs (dict): The inputs and targets of the model.

        Returns:
            torch.FloatTensor: The loss value.
        """
        # # Get outputs, predictions and prediction loss
        # outputs = model(**inputs)
        # pred_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

        # Call prediction_step
        pred_loss, preds, refs = self.prediction_step_custom(model, inputs, prediction_loss_only=False)
        
        # Post-process the predictions and references
        ## In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        ## Replace -100s in the labels as we can't decode them
        refs = torch.where(refs != -100, refs, torch.tensor(self.tokenizer.pad_token_id).to(DEVICE))
        decoded_refs = self.tokenizer.batch_decode(refs, skip_special_tokens=True)

        ## Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_refs = [ref.strip() for ref in decoded_refs]

        # Calculate the classifier-guided loss
        classifier_loss = self.classifier_guided_loss(decoded_preds, 0)

        # Calculate total loss as a weighted sum of the classifier loss and the prediction loss
        total_loss = 0.5 * classifier_loss + 0.5 * pred_loss

        # Return total loss
        return total_loss
    

def get_indices(dataset):
    """
    Saves the indices of data that is to_neutral and to_toxic.
    """
    to_neutral_idx = []
    to_toxic_idx = []
    for i in range(len(dataset)):
        if dataset[i]["source"].startswith("to_neutral"):
            to_neutral_idx.append(i)
        else:
            to_toxic_idx.append(i)

    return to_neutral_idx, to_toxic_idx

def compute_metrics_bd(eval_preds, tokenizer, bd_dataset, shuffled_data=False):
    """
    Function to calculate the metrics for trainer.evaluate().
    This function is for the bi-directional model.
    
    Args:
        eval_preds (tuple): Tuple containing the predictions and references
        tokenizer (PreTrainedTokenizer): tokenizer to use for decoding the predictions
        shuffled_data (bool): Whether the data is shuffled or not
        bd_dataset (DatasetDict): Bidirectional dataset created using create_bidirectional_datasets e.g., raw_datasets_bd["validation"]

    Returns:
        dict: Dictionary containing the metrics
    """
    preds, refs = eval_preds

    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    refs = np.where(refs != -100, refs, tokenizer.pad_token_id)
    decoded_refs = tokenizer.batch_decode(refs, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_refs = [ref.strip() for ref in decoded_refs]
    
    # If shuffled data is false, have to_neutral_preds and to_neutral_refs just be predictions and refs with even indices
    if not shuffled_data:
        to_neutral_preds = decoded_preds[::2]
        to_neutral_refs = decoded_refs[::2]
    # Otherwise, get the indices to use when splitting predictions and refs to to_neutral and to_toxic
    else:
        # Get the indices to use when splitting predictions and refs to to_neutral and to_toxic
        to_neutral_idx = get_indices(bd_dataset)[0]

        # Retrieve based on the indices
        to_neutral_preds = [decoded_preds[i] for i in to_neutral_idx]
        to_neutral_refs = [decoded_refs[i] for i in to_neutral_idx]
    
    # Evaluate metrics for to_neutral
    to_neutral_metrics = evaluate_metrics(
        to_neutral_refs,
        to_neutral_preds,
        to_neutral=True
    )

    # Return dictionary of to_neutral metrics
    return to_neutral_metrics

class Seq2SeqTrainingArgumentsCustomLoss(Seq2SeqTrainingArguments):
    def __init__(self, classifier_loss_weight, prediction_loss_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier_loss_weight = classifier_loss_weight
        self.prediction_loss_weight = prediction_loss_weight

class Seq2SeqTrainerCustomLoss(Seq2SeqTrainer):
    def __init__(self, model_toxicity, tokenizer_toxicity, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier_loss_weight = self.args.classifier_loss_weight
        self.prediction_loss_weight = self.args.prediction_loss_weight
        self.model_toxicity = model_toxicity.to(self.model.device)
        self.tokenizer_toxicity = tokenizer_toxicity
        self.loss_fct = torch.nn.CrossEntropyLoss().to(self.model.device)

    def generate_tokens_custom(
            self,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            **gen_kwargs,
        ):        
        # Prepare the inputs
        inputs = self._prepare_inputs(inputs)
        
        # Merge default generation kwargs with any user-provided kwargs
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
            
        # Remove 'num_beams' and 'max_length' from gen_kwargs if they are None
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")
        
        # Set 'synced_gpus' to False in gen_kwargs
        gen_kwargs["synced_gpus"] = False
        
        # Copy inputs to a new dictionary for generation
        generation_inputs = inputs.copy()
        
        # Remove 'decoder_input_ids' if it was created from 'labels'
        if ("labels" in generation_inputs and 
            "decoder_input_ids" in generation_inputs and 
            generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape):
            generation_inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
        
        # Generate tokens using the model
        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)
        
        # Hack to ensure generation config is not initialized for each iteration (temporary)
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False
        
        # Retrieve the GenerationConfig from the model
        gen_config = self.model.generation_config
        
        # Pad generated tokens if they are shorter than max length
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        
        return generated_tokens

    def classifier_guided_loss(self, generated_text, target_labels):
        # Tokenize the generated text
        preds_tokenized = self.tokenizer_toxicity(generated_text, return_tensors="pt", padding=True)
        
        # Move inputs to the same device as the toxicity model
        preds_tokenized = {key: tensor.to(self.model_toxicity.device) for key, tensor in preds_tokenized.items()}
        
        # Get the classifier's output logits
        logits = self.model_toxicity(**preds_tokenized)["logits"]
        
        # Calculate the cross-entropy loss
        loss = self.loss_fct(logits, target_labels)
        
        return loss

    def compute_loss(self, model, inputs):
        # Get outputs, predictions and prediction loss
        outputs = model(**inputs)
        pred_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]

        # Call prediction_step
        preds, refs = self.prediction_step_custom(inputs)

        # Decode predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        
        # Decode references
        refs = torch.where(refs != -100, refs, torch.tensor(self.tokenizer.pad_token_id).to(DEVICE))
        decoded_refs = self.tokenizer.batch_decode(refs, skip_special_tokens=True)
        decoded_refs = [ref.strip() for ref in decoded_refs]

        # Identify indices corresponding to <to_neutral> and <to_toxic>
        decoded_inputs = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        to_neutral_idx = [i for i, decoded_input in enumerate(decoded_inputs) if decoded_input.startswith("to_neutral")]
        to_toxic_idx = [i for i, decoded_input in enumerate(decoded_inputs) if decoded_input.startswith("to_toxic")]
        
        # Calculate target labels based on indices. The length should be the same as the number of predictions
        target_labels = torch.zeros(len(decoded_preds), dtype=torch.long, device=DEVICE)
        target_labels[to_neutral_idx] = 0
        target_labels[to_toxic_idx] = 1
        
        # Calculate the classifier-guided loss
        classifier_loss = self.classifier_guided_loss(decoded_preds, target_labels)

        # Calculate total loss as a weighted sum of the classifier loss and the prediction loss
        total_loss = self.classifier_loss_weight * classifier_loss + self.prediction_loss_weight * pred_loss
        print(f"Classifier-guided loss: {classifier_loss}, Prediction loss: {pred_loss}, Total loss: {total_loss}")

        # Return total loss
        return total_loss
    
class Seq2SeqTrainingArgumentsCustomLoss(Seq2SeqTrainingArguments):
    def __init__(self, classifier_loss_weight, prediction_loss_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier_loss_weight = classifier_loss_weight
        self.prediction_loss_weight = prediction_loss_weight

class Seq2SeqTrainerCustomLoss(Seq2SeqTrainer):
    def __init__(self, model_toxicity, tokenizer_toxicity, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier_loss_weight = self.args.classifier_loss_weight
        self.prediction_loss_weight = self.args.prediction_loss_weight
        self.model_toxicity = model_toxicity.to(self.model.device)
        self.tokenizer_toxicity = tokenizer_toxicity
        self.loss_fct = torch.nn.CrossEntropyLoss().to(self.model.device)

    def generate_pred_tokens(
            self,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            **gen_kwargs,
        ):        
        # Prepare the inputs
        inputs = self._prepare_inputs(inputs)
        
        # Merge default generation kwargs with any user-provided kwargs
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
            
        # # Remove 'num_beams' and 'max_length' from gen_kwargs if they are None
        # if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
        #     gen_kwargs.pop("num_beams")
        # if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
        #     gen_kwargs.pop("max_length")
        
        # Set 'synced_gpus' to False in gen_kwargs
        gen_kwargs["synced_gpus"] = False
        
        # Copy inputs to a new dictionary for generation
        generation_inputs = inputs.copy()
        
        # Remove 'decoder_input_ids' if it was created from 'labels'
        if ("labels" in generation_inputs and 
            "decoder_input_ids" in generation_inputs and 
            generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape):
            generation_inputs = {k: v for k, v in inputs.items() if k != "decoder_input_ids"}
        
        # Generate tokens using the model
        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)
        
        # # Hack to ensure generation config is not initialized for each iteration (temporary)
        # if self.model.generation_config._from_model_config:
        #     self.model.generation_config._from_model_config = False
        
        # # Retrieve the GenerationConfig from the model
        # gen_config = self.model.generation_config
        
        # # Pad generated tokens if they are shorter than max length
        # if generated_tokens.shape[-1] < gen_config.max_length:
        #     generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        
        return generated_tokens

    def classifier_guided_loss(self, generated_text, target_labels):
        # Tokenize the generated text
        preds_tokenized = self.tokenizer_toxicity(generated_text, return_tensors="pt", padding=True)
        
        # Move inputs to the same device as the toxicity model
        preds_tokenized = {key: tensor.to(self.model_toxicity.device) for key, tensor in preds_tokenized.items()}
        
        # Get the classifier's output logits
        logits = self.model_toxicity(**preds_tokenized)["logits"]
        
        # Calculate the cross-entropy loss
        loss = self.loss_fct(logits, target_labels)
        
        return loss

    def compute_loss(self, model, inputs):
        # start_time = timer()

        # Get outputs, predictions and prediction loss
        outputs = model(**inputs)
        pred_loss = outputs.loss if isinstance(outputs, dict) else outputs[0]
        # pred_loss_time = timer()
        # print(f"Time to compute prediction loss: {pred_loss_time - start_time:.4f} seconds")

        # Call prediction_step
        preds = self.generate_pred_tokens(inputs)
        # preds_time = timer()
        # print(f"Time to generate predictions: {preds_time - pred_loss_time:.4f} seconds")

        # Decode predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_preds = [pred.strip() for pred in decoded_preds]
        # decoding_time = timer()
        # print(f"Time to decode predictions: {decoding_time - preds_time:.4f} seconds")

        # Identify indices corresponding to <to_neutral> and <to_toxic>
        decoded_inputs = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        # to_neutral_idx = [i for i, decoded_input in enumerate(decoded_inputs) if decoded_input.startswith("to_neutral")]
        # to_toxic_idx = [i for i, decoded_input in enumerate(decoded_inputs) if decoded_input.startswith("to_toxic")]

        # Convert decoded_inputs to a NumPy array
        decoded_inputs_np = np.array(decoded_inputs)

        # Vectorized operation to identify indices
        to_neutral_mask = np.char.startswith(decoded_inputs_np, "to_neutral")
        to_toxic_mask = np.char.startswith(decoded_inputs_np, "to_toxic")

        # Get the indices where the conditions are True
        to_neutral_idx = np.where(to_neutral_mask)[0].tolist()
        to_toxic_idx = np.where(to_toxic_mask)[0].tolist()
        # indices_time = timer()
        # print(f"Time to identify indices: {indices_time - decoding_time:.4f} seconds")

        # Calculate target labels based on indices. The length should be the same as the number of predictions
        target_labels = torch.zeros(len(decoded_preds), dtype=torch.long, device=self.model_toxicity.device)
        target_labels[to_neutral_idx] = 0
        target_labels[to_toxic_idx] = 1
        # target_labels_time = timer()
        # print(f"Time to calculate target labels: {target_labels_time - indices_time:.4f} seconds")

        # Debug print statement to check first 10 decoded inputs and target labels
        # for i in range(10):
        #     print(f"Decoded input: {decoded_inputs[i]}, Target label: {target_labels[i]}")

        # Calculate the classifier-guided loss
        classifier_loss = self.classifier_guided_loss(decoded_preds, target_labels)
        # classifier_loss_time = timer()
        # print(f"Time to compute classifier-guided loss: {classifier_loss_time - target_labels_time:.4f} seconds")

        # Calculate total loss as a weighted sum of the classifier loss and the prediction loss
        total_loss = self.classifier_loss_weight * classifier_loss + self.prediction_loss_weight * pred_loss
        # total_loss_time = timer()
        # print(f"Time to compute total loss: {total_loss_time - classifier_loss_time:.4f} seconds")

        # Return total loss
        return total_loss
    
# Define objective function for Optuna
def compute_objective(metrics):
    return metrics["eval_Overall"]

# Define the hyperparameter search space for Optuna
def optuna_hp_space(trial):
    return {
        "classifier_loss_weight": trial.suggest_float("classifier_loss_weight", 0.0, 1.0, step=0.1),
    }

def model_init(trial):
    return T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)

def setup_trainer_customloss_optuna(output_dir_name,
                train_dataset,
                eval_dataset,
                model_checkpoint="t5-small",
                per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                num_train_epochs=NUM_TRAIN_EPOCHS,
                max_length=MAX_OUTPUT_LENGTH,
                num_beams=NUM_BEAMS,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                classifier_loss_weight=0.5,
                bidirectional=False,
                ):
    
    # Instantiate model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint).to(DEVICE)
    tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)

    # Define the data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model, return_tensors="pt", padding=True)

    # Define generation config
    generation_config = GenerationConfig(
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        eos_token_id=model.config.eos_token_id,
        bos_token_id=model.config.bos_token_id,
        pad_token_id=model.config.pad_token_id,
        decoder_start_token_id=model.config.pad_token_id
        )

    # Save the generation config
    gen_config_path = f"../models/{output_dir_name}/generation_config"
    generation_config.save_pretrained(gen_config_path)

    # Define the training arguments
    args = Seq2SeqTrainingArgumentsCustomLoss(
        output_dir=f'../models/{output_dir_name}',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate, 
        predict_with_generate=True,
        generation_config=gen_config_path,
        fp16=True,
        report_to="wandb",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="Overall",
        greater_is_better=True,
        generation_max_length=max_length,
        classifier_loss_weight=classifier_loss_weight,
        bidirectional=bidirectional,
    )
    
    # Instantiate the trainer
    trainer = Seq2SeqTrainerCustomLoss(
        model=model,
        tokenizer=tokenizer,
        model_toxicity=model_toxicity,
        tokenizer_toxicity=tokenizer_toxicity,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        model_init=model_init
    )

    return trainer

# Create a sample of the raw datasets for hyperparameter optimization
raw_datasets_sample = raw_datasets.copy()
raw_datasets_sample["train"] = raw_datasets_sample["train"].shuffle(seed=RANDOM_SEED).select(range(2000))

# Preprocess the sample
prefixed_datasets_sample = add_prefix(raw_datasets_sample)

tokenized_datasets_sample_t5_small = prefixed_datasets_sample.map(
    preprocess_function,
    fn_kwargs={'tokenizer': tokenizer_t5_small},
    batched=True,
    remove_columns=["source", "target"],
)

# Setup the trainer
trainer_sample_t5_small_cl = setup_trainer_customloss_optuna(
    output_dir_name="t5-small-detoxify-cl-optuna",
    model_checkpoint="t5-small",
    train_dataset=tokenized_datasets_sample_t5_small["train"],
    eval_dataset=tokenized_datasets_sample_t5_small["validation"],
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    max_length=MAX_OUTPUT_LENGTH,
    num_beams=NUM_BEAMS,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    classifier_loss_weight=0.5,
    bidirectional=False,
)

# Run hyperparameter search
best_trial_sample_t5_small_cl = trainer_sample_t5_small_cl.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=10,
    study_name="optuna_t5_small_detoxify_cl",
    compute_objective=compute_objective,
    pruner=optuna.pruners.HyperbandPruner(), # Recommended here: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.htm
    sampler=optuna.samplers.TPESampler(),
)