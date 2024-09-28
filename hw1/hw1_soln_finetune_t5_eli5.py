import datasets
from datasets import load_dataset
from evaluate import load
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

import nltk
import numpy as np
import random
import pandas as pd
# from IPython.display import display, HTML

nltk.download('punkt')

raw_datasets = load_dataset("rexarski/eli5_category", trust_remote_code=True)

print("Dataset columns:", raw_datasets['train'].column_names)

def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = random.sample(range(len(dataset)), num_examples)
    df = pd.DataFrame(dataset[picks])
    print(df)
    # display(HTML(df.to_html()))

if "validation" not in raw_datasets:
    raw_datasets = raw_datasets['train'].train_test_split(test_size=0.1)
    raw_datasets['validation'] = raw_datasets['test']
    del raw_datasets['test']

model_checkpoint = "t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

if model_checkpoint.startswith("t5"):
    prefix = "answer: "
else:
    prefix = ""

max_input_length = 512
max_target_length = 128

show_random_elements(raw_datasets['train'], num_examples=3)

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
    
    inputs = [prefix + "Question: " + q + " Category: " + c for q, c in zip(questions, categories)]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    
    answers_list = examples["answers"]
    
    labels_text = []
    for ans in answers_list:
        if 'text' in ans and isinstance(ans['text'], list) and len(ans['text']) > 0:
            labels_text.append(ans['text'][0])
        else:
            labels_text.append("")
    
    labels = tokenizer(text_target=labels_text, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=raw_datasets['train'].column_names,
    load_from_cache_file=False  # Disable caching
)

show_random_elements(tokenized_datasets['train'], num_examples=3)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


batch_size = 16

args = Seq2SeqTrainingArguments(
    output_dir=f"{model_checkpoint}-finetuned-eli5_category",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size, 
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=False, 
    push_to_hub=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

metric = load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]
    
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {key: value * 100 for key, value in result.items()}
    
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

