## Fine-tune T5-small
tokenized_datasets_t5_small = add_prefix(raw_datasets).map(
    preprocess_function,
    fn_kwargs={'tokenizer': tokenizer_t5_small},
    batched=True,
    remove_columns=["source", "target"],
)

trainer_t5_small = setup_trainer(
    output_dir_name="t5-small-detoxify",
    model_checkpoint="t5-small",
    train_dataset=tokenized_datasets_t5_small["train"],
    eval_dataset=tokenized_datasets_t5_small["validation"],
)

wandb.init(project="w266_final_project", name="t5-small-detoxify")
trainer_t5_small.train()
wandb.finish()

print(f"Best model checkpoint: {trainer_t5_small.state.best_model_checkpoint}")

## Fine-tune T5-small (bidirectional, no shuffle)
tokenized_datasets_bd_noshuffle_t5_small = create_bidirectional_dataset(raw_datasets, shuffle=False).map(
    preprocess_function,
    fn_kwargs={'tokenizer': tokenizer_t5_small},
    batched=True,
    remove_columns=["source", "target"],
)

trainer_t5_small_bd_noshuffle = setup_trainer(
    output_dir_name="t5-small-detoxify-bd-noshuffle",
    model_checkpoint="t5-small",
    train_dataset=tokenized_datasets_bd_noshuffle_t5_small["train"],
    eval_dataset=tokenized_datasets_bd_noshuffle_t5_small["validation"],
    compute_metrics=partial(compute_metrics_bd, bd_dataset=raw_datasets_bd_noshuffle["validation"], shuffled_data=False)
    )

wandb.init(project="w266_final_project", name="t5-small-bd-noshuffle-detoxify")
trainer_t5_small_bd_noshuffle.train()
wandb.finish()

print(f"Best checkpoint path: {trainer_t5_small_bd_noshuffle.state.best_model_checkpoint}")

## Fine-tune T5-small (bidirectional, shuffle)
tokenized_datasets_bd_shuffle_t5_small = create_bidirectional_dataset(raw_datasets, shuffle=True).map(
    preprocess_function,
    fn_kwargs={'tokenizer': tokenizer_t5_small},
    batched=True,
    remove_columns=["source", "target"],
)

trainer_t5_small_bd_shuffle = setup_trainer(
    output_dir_name="t5-small-detoxify-bd-shuffle",
    model_checkpoint="t5-small",
    train_dataset=tokenized_datasets_bd_shuffle_t5_small["train"],
    eval_dataset=tokenized_datasets_bd_shuffle_t5_small["validation"],
    compute_metrics=partial(compute_metrics_bd, bd_dataset=raw_datasets_bd_shuffle["validation"], shuffled_data=True)
    )

wandb.init(project="w266_final_project", name="t5-small-bd-shuffle-detoxify")
trainer_t5_small_bd_shuffle.train()
wandb.finish()

print(f"Best checkpoint path: {trainer_t5_small_bd_shuffle.state.best_model_checkpoint}")

## Fine-tune T5-small (unidirectional, augmented, all filters)
tokenized_datasets_all_filters_t5_small = add_prefix(aug_datasets_all_filters).map(
    preprocess_function,
    fn_kwargs={'tokenizer': tokenizer_t5_small},
    batched=True,
    remove_columns=["source", "target"],
)

trainer_t5_small_aug = setup_trainer(
    output_dir_name="t5-small-detoxify-aug-all-filters",
    model_checkpoint="t5-small",
    train_dataset=tokenized_datasets_all_filters_t5_small["train"],
    eval_dataset=tokenized_datasets_all_filters_t5_small["validation"],
)

wandb.init(project="w266_final_project", name="t5-small-detoxify-aug-all-filters")
trainer_t5_small_aug.train()
wandb.finish()

# Print path to the best checkpoint
print(trainer_t5_small_aug.state.best_model_checkpoint)

## Fine-tune T5-small (unidirectional, augmented, no acceptability filter)
tokenized_datasets_no_acceptability_filter_t5_small = add_prefix(aug_datasets_no_acceptability_filter).map(
    preprocess_function,
    fn_kwargs={'tokenizer': tokenizer_t5_small},
    batched=True,
    remove_columns=["source", "target"],
)

trainer_t5_small_aug_no_acceptability_filter = setup_trainer(
    output_dir_name="t5-small-detoxify-aug-no-acceptability-filter",
    model_checkpoint="t5-small",
    train_dataset=tokenized_datasets_no_acceptability_filter_t5_small["train"],
    eval_dataset=tokenized_datasets_no_acceptability_filter_t5_small["validation"],
)

wandb.init(project="w266_final_project", name="t5-small-detoxify-aug-no-acceptability-filter")
trainer_t5_small_aug_no_acceptability_filter.train()
wandb.finish()

# Print path to the best checkpoint
print(f"Best model checkpoint: {trainer_t5_small_aug_no_acceptability_filter.state.best_model_checkpoint}")

## Fine-tune T5-small (unidirectional, augmented, no similarity filter)
tokenized_datasets_no_similarity_filter_t5_small = add_prefix(aug_datasets_no_similarity_filter).map(
    preprocess_function,
    fn_kwargs={'tokenizer': tokenizer_t5_small},
    batched=True,
    remove_columns=["source", "target"],
)

trainer_t5_small_aug_no_similarity_filter = setup_trainer(
    output_dir_name="t5-small-detoxify-aug-no-similarity-filter",
    model_checkpoint="t5-small",
    train_dataset=tokenized_datasets_no_similarity_filter_t5_small["train"],
    eval_dataset=tokenized_datasets_no_similarity_filter_t5_small["validation"],
)

wandb.init(project="w266_final_project", name="t5-small-detoxify-aug-no-similarity-filter")
trainer_t5_small_aug_no_similarity_filter.train()
wandb.finish()

print(f"Best model checkpoint: {trainer_t5_small_aug_no_similarity_filter.state.best_model_checkpoint}")

## Fine-tune T5-small (unidirectional, augmented, no toxicity filter)
tokenized_datasets_no_toxicity_filter_t5_small = add_prefix(aug_datasets_no_toxicity_filter).map(
    preprocess_function,
    fn_kwargs={'tokenizer': tokenizer_t5_small},
    batched=True,
    remove_columns=["source", "target"],
)

trainer_t5_small_aug_no_toxicity_filter = setup_trainer(
    output_dir_name="t5-small-detoxify-aug-no-toxicity-filter",
    model_checkpoint="t5-small",
    train_dataset=tokenized_datasets_no_toxicity_filter_t5_small["train"],
    eval_dataset=tokenized_datasets_no_toxicity_filter_t5_small["validation"],
)

wandb.init(project="w266_final_project", name="t5-small-detoxify-aug-no-toxicity-filter")
trainer_t5_small_aug_no_toxicity_filter.train()
wandb.finish()

print(f"Best model checkpoint: {trainer_t5_small_aug_no_toxicity_filter.state.best_model_checkpoint}")