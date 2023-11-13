def add_prefix(datasetdict, prefix="to_neutral: "):
    """Adds a prefix to the source sequence in the dataset."""
    datasetdict_copy = datasetdict.copy()
    datasetdict_copy["train"] = datasetdict_copy["train"].map(lambda x: {"source": prefix + x["source"]})
    datasetdict_copy["validation"] = datasetdict_copy["validation"].map(lambda x: {"source": prefix + x["source"]})
    datasetdict_copy["test"] = datasetdict_copy["test"].map(lambda x: {"source": prefix + x["source"]})
    datasetdict_copy = DatasetDict(datasetdict_copy)
    return datasetdict_copy

def create_bidirectional_dataset(datasets, shuffle=True):
    """
    Creates a bi-directional dataset from the original dataset.

    Args:
        datasets (DatasetDict): DatasetDict object containing the original dataset.
        shuffle (bool): Whether to shuffle the dataset or not.
    
    Returns:
        extended_datasets (DatasetDict): DatasetDict object containing the bi-directional dataset.
    """

    def bidirectional_extension(dataset):
        new_data = {
            "source": [],
            "target": []
        }
        for src, tgt in zip(dataset['source'], dataset['target']):
            new_data['source'].extend([f'to_neutral: {src}', f'to_toxic: {tgt}'])
            new_data['target'].extend([tgt, src])
        return new_data

    extended_train_data = bidirectional_extension(datasets["train"])
    extended_validation_data = bidirectional_extension(datasets["validation"])
    extended_test_data = bidirectional_extension(datasets["test"])

    extended_datasets = DatasetDict({
        "train": Dataset.from_dict(extended_train_data),
        "validation": Dataset.from_dict(extended_validation_data),
        "test": Dataset.from_dict(extended_test_data)
    })

    if shuffle:
        extended_datasets["train"] = extended_datasets["train"].shuffle(seed=RANDOM_SEED)
        
    return extended_datasets

def preprocess_dataset(dataset, tokenizer):
    """Preprocesses a dataset using a tokenizer."""
    def preprocess_function(examples, tokenizer):
        """Preprocess function for T5."""
        model_inputs = tokenizer(
            examples["source"],
            text_target=examples["target"],
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        )
        return model_inputs

    return dataset.map(
        preprocess_function,
        fn_kwargs={'tokenizer': tokenizer},
        batched=True,
        remove_columns=["source", "target"],
    )

def post_process(preds, refs, tokenizer):
    """
    Post-process function for T5.

    Args:
        preds (list): list of predicted sequences
        refs (list): list of reference sequences
        tokenizer (PreTrainedTokenizer): tokenizer to use for decoding

    Returns:
        decoded_preds (list): list of decoded predicted sequences
        decoded_refs (list): list of decoded reference sequences
    """
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

    return decoded_preds, decoded_refs

def compute_metrics(eval_preds, tokenizer):
    """
    Function to calculate the metrics for trainer.evaluate().

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer to use for decoding the predictions
        eval_preds (tuple): Tuple containing the predictions and references

    Returns:
        dict: Dictionary containing the metrics
    """
    preds, refs = eval_preds

    # Post-process the predictions and references
    decoded_preds, decoded_refs = post_process(preds, refs, tokenizer)
    
    # Evaluate metrics
    return evaluate_metrics(
        decoded_refs,
        decoded_preds,
        tokenizer_toxicity=tokenizer_toxicity,
        model_toxicity=model_toxicity,
        tokenizer_acceptability=tokenizer_acceptability,
        model_acceptability=model_acceptability,
        include_bleurt=False
    )

def compute_metrics_bd(eval_preds, tokenizer, bd_dataset, shuffled_data=False):
    """
    Function to calculate the metrics for trainer.evaluate().
    This function is for the bi-directional model.
    
    Args:
        eval_preds (tuple): Tuple containing the predictions and references
        tokenizer (PreTrainedTokenizer): tokenizer to use for decoding the predictions
        shuffled_data (bool): Whether the data is shuffled or not
        bd_dataset (DatasetDict): Bidirectional dataset to use for testing created using create_bidirectional_datasets
                                  For example, raw_datasets_bd["validation"] or raw_datasets_bd["test"]

    Returns:
        dict: Dictionary containing the metrics
    """
    preds, refs = eval_preds

    # Post-process the predictions and references
    decoded_preds, decoded_refs = post_process(preds, refs, tokenizer)
    
    # If shuffled data is false, have to_neutral_preds and to_neutral_refs just be predictions and refs with even indices
    if not shuffled_data:
        to_neutral_preds = decoded_preds[::2]
        to_neutral_refs = decoded_refs[::2]
    # Otherwise, get the indices to use when splitting predictions and refs to to_neutral and to_toxic
    else:
        # Get the indices to use when splitting predictions and refs to to_neutral and to_toxic
        to_neutral_idx = [i for i, input_sentence in enumerate(bd_dataset['source']) if input_sentence.startswith("to_neutral")]

        # Retrieve based on the indices
        to_neutral_preds = [decoded_preds[i] for i in to_neutral_idx]
        to_neutral_refs = [decoded_refs[i] for i in to_neutral_idx]
    
    # Evaluate metrics for to_neutral
    to_neutral_metrics = evaluate_metrics(
        to_neutral_refs,
        to_neutral_preds,
    )

    # Return dictionary of to_neutral metrics
    return to_neutral_metrics

def setup_trainer(output_dir_name,
                train_dataset,
                eval_dataset,
                compute_metrics,
                model_checkpoint="t5-small",
                per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                num_train_epochs=NUM_TRAIN_EPOCHS,
                max_length=MAX_OUTPUT_LENGTH,
                num_beams=NUM_BEAMS,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                report_to="wandb",
                ):
    """
    Set up a Seq2SeqTrainer object for training a T5 model.

    Default parameters based on this: https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/models/hf_model.py#L55

    Args:
        output_dir_name (str): What to name the model in the output directory.
        train_dataset (Dataset): Training dataset.
        eval_dataset (Dataset): Evaluation dataset.
        compute_metrics (function): Function to compute metrics. Change this to compute_metrics_bd if using a bi-directional model.
        model_checkpoint (str): Model checkpoint to use.
        per_device_train_batch_size (int): Batch size for training.
        per_device_eval_batch_size (int): Batch size for evaluation.
        learning_rate (float): Learning rate.
        num_train_epochs (int): Number of training epochs.
        max_length (int): Maximum length of the output sequence.
        num_beams (int): Number of beams for beam search.
        early_stopping_patience (int): Number of epochs to wait before early stopping.
        report_to (str): Where to report results to. Either "wandb" or "none".

    Returns:
        Seq2SeqTrainer: Trainer object for training the T5 model.
    """
    
    # Instantiate model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
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
    args = Seq2SeqTrainingArguments(
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
        report_to=report_to,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="Overall",
        greater_is_better=True,
        generation_max_length=max_length,
    )
   
    # Instantiate the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]

    )

    return trainer

def setup_trainer(output_dir_name,
                train_dataset,
                eval_dataset,
                compute_metrics,
                model_checkpoint="t5-small",
                per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                num_train_epochs=NUM_TRAIN_EPOCHS,
                max_length=MAX_OUTPUT_LENGTH,
                num_beams=NUM_BEAMS,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                report_to="wandb",
                ):
    """
    Set up a Seq2SeqTrainer object for training a T5 model.

    Default parameters based on this: https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/models/hf_model.py#L55

    Args:
        output_dir_name (str): What to name the model in the output directory.
        train_dataset (Dataset): Training dataset.
        eval_dataset (Dataset): Evaluation dataset.
        compute_metrics (function): Function to compute metrics. Change this to compute_metrics_bd if using a bi-directional model.
        model_checkpoint (str): Model checkpoint to use.
        per_device_train_batch_size (int): Batch size for training.
        per_device_eval_batch_size (int): Batch size for evaluation.
        learning_rate (float): Learning rate.
        num_train_epochs (int): Number of training epochs.
        max_length (int): Maximum length of the output sequence.
        num_beams (int): Number of beams for beam search.
        early_stopping_patience (int): Number of epochs to wait before early stopping.
        report_to (str): Where to report results to. Either "wandb" or "none".

    Returns:
        Seq2SeqTrainer: Trainer object for training the T5 model.
    """
    
    # Instantiate model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
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
    args = Seq2SeqTrainingArguments(
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
        report_to=report_to,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="Overall",
        greater_is_better=True,
        generation_max_length=max_length,
    )
   
    # Instantiate the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]

    )

    return trainer

def training_pipeline(model_name, project_name="t5-detox", model_checkpoint="t5-small", use_validation=True, raw_datasets=raw_datasets, bidirectional=False, shuffle=False, do_train=True):
    """
    Pipeline for training a T5 model. Saves the best model checkpoint to a txt file. Can also be used for evaluating a model (use test set instead of validation set).

    Args:
        model_name (str): Name of the model to name the output directory and wandb run.
        project_name (str): Name of the wandb project.
        model_checkpoint (str): Model checkpoint to use.
        use_validation (bool): Whether to use the validation set or not.
        raw_datasets (DatasetDict): DatasetDict object containing the original dataset.
        bidirectional (bool): Whether to use a bi-directional model or not.
        shuffle (bool): Whether to shuffle the dataset or not.
        do_train (bool): Whether to train the model or not.

    Returns:
        trainer (Seq2SeqTrainer): Trainer object for training the T5 model.
    """
    
    # Preprocess dataset (add prefixes / make bidirectional)
    if bidirectional:
        raw_datasets = create_bidirectional_dataset(raw_datasets, shuffle=shuffle)
    else:
        raw_datasets = add_prefix(raw_datasets)

    # Tokenize dataset
    tokenized_datasets = preprocess_dataset(raw_datasets, tokenizer_t5_small)

    # Define compute_metrics function depending on bidirectional or not
    if bidirectional and use_validation:
        bd_dataset = raw_datasets["validation"]
    elif bidirectional and not use_validation:
        bd_dataset = raw_datasets["test"]
    else:
        bd_dataset = None

    compute_metrics_fn = partial(compute_metrics_bd, bd_dataset=bd_dataset, shuffled_data=shuffle) if bd_dataset else compute_metrics

    # Setup trainer
    trainer = setup_trainer(
        output_dir_name=model_name,
        model_checkpoint=model_checkpoint,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"] if use_validation else tokenized_datasets["test"],
        compute_metrics=compute_metrics_fn
    )

    if do_train:
        # Initialize wandb
        wandb.init(project=project_name, name=model_name)
        trainer.train()
        wandb.finish()

        # Get the best checkpoint path for the model
        checkpoint_path = trainer.state.best_model_checkpoint

        # Save the checkpoint path for the best model
        with open(BEST_MODEL_CHECKPOINT_PATH, "a") as file:
            file.write(f"{model_name}: {checkpoint_path}\n")

    return trainer