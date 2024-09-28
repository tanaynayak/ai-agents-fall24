import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from evaluate import load
import torch
from tqdm.auto import tqdm


import nltk
nltk.download('punkt')


model_checkpoint = "t5-base-finetuned-eli5_category/checkpoint-15489"  # Change to your checkpoint


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to("cuda")


raw_datasets = load_dataset("rexarski/eli5_category")


def preprocess_function(examples):
    titles = examples['title']
    selftexts = examples['selftext']
    categories = examples['category']
    
    questions = []
    for title, selftext in zip(titles, selftexts):
        if selftext and selftext.strip():
            question = title.strip() + " " + selftext.strip()
        else:
            question = title.strip()
        questions.append(question)
    
    inputs = [f"answer: Question: {q} Category: {c}" for q, c in zip(questions, categories)]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    

    answers_list = examples["answers"]
    labels_text = [ans['text'][0] if 'text' in ans and isinstance(ans['text'], list) and len(ans['text']) > 0 else "" for ans in answers_list]
    
    labels = tokenizer(text_target=labels_text, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs


tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets['train'].column_names
)


metric_rouge = load("rouge")
metric_bleu = load("sacrebleu")
metric_bertscore = load("bertscore")


def generate_predictions(batch):
    inputs = batch["input_ids"].to("cuda")
    attention_mask = batch["attention_mask"].to("cuda")


    generated_ids = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=128,
        early_stopping=True
    )
    

    decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded_preds


def clean_labels(labels):
    """
    Ensure labels are within a valid range and convert invalid values like -100 to padding token id.
    """

    labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
    return labels


def compute_metrics(decoded_preds, decoded_labels):
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    

    rouge_result = metric_rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rouge_result = {key: value * 100 for key, value in rouge_result.items()}
    

    bleu_result = metric_bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    

    bertscore_result = metric_bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
    bertscore_avg = {
        'precision': sum(bertscore_result['precision']) / len(bertscore_result['precision']),
        'recall': sum(bertscore_result['recall']) / len(bertscore_result['recall']),
        'f1': sum(bertscore_result['f1']) / len(bertscore_result['f1']),
    }
    

    final_result = {
        **rouge_result,
        "bleu": bleu_result["score"],
        "bertscore_precision": bertscore_avg["precision"] * 100,
        "bertscore_recall": bertscore_avg["recall"] * 100,
        "bertscore_f1": bertscore_avg["f1"] * 100,
    }
    
    return final_result


def evaluate_with_default_parameters(num_samples=None):
    """
    Evaluate the model using batch processing with default generation parameters for a limited number of samples.
    Using GPU for efficient processing.
    
    Args:
        num_samples (int): The number of samples from the validation set to evaluate on. If None, evaluate on the full set.
    """
    

    validation_dataset = tokenized_datasets["validation1"]
    

    if num_samples is not None:
        validation_dataset = validation_dataset.select(range(min(num_samples, len(validation_dataset))))
    

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=8, collate_fn=data_collator)

    all_preds = []
    all_labels = []
    

    for batch in tqdm(validation_loader):
        # Generate predictions
        preds = generate_predictions(batch)
        
        # Clean and decode labels (ground-truth)
        cleaned_labels = clean_labels(batch["labels"])  # Ensure the labels are valid
        labels = tokenizer.batch_decode(cleaned_labels, skip_special_tokens=True)
        
        all_preds.extend(preds)
        all_labels.extend(labels)
    

    return compute_metrics(all_preds, all_labels)


num_samples_to_evaluate = 500  


print(f"Evaluating with default generation parameters on {num_samples_to_evaluate} samples...")
metrics = evaluate_with_default_parameters(num_samples=num_samples_to_evaluate)


df_results = pd.DataFrame([metrics])
print(df_results.to_string())