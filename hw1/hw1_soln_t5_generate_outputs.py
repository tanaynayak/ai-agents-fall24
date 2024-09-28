import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import random

model_checkpoint = "t5-base-finetuned-eli5_category/checkpoint-15489"  
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Load the ELI5 dataset and validation split
raw_datasets = load_dataset("rexarski/eli5_category")
validation_set = raw_datasets['validation1']

# Sample 5 random questions from the validation set
def sample_random_questions(dataset, num_samples=5):
    indices = random.sample(range(len(dataset)), num_samples)
    return [dataset[i] for i in indices]

def generate_answer(question):
    
    inputs = tokenizer(f"answer: Question: {question}", return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]  
    output = model.generate(
        input_ids=input_ids, 
        max_length=256, 
        num_beams=4, 
        top_k=20, 
        top_p=1.0, 
        temperature=1.0,
        do_sample=True
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)


sampled_questions = sample_random_questions(validation_set)

for example in sampled_questions:
    question = example['title']  
    generated_answer = generate_answer(question)
    
    print(f"Question: {question}")
    
  
    print(f"Generated Answer: {generated_answer}")
    
 
    print(f"Answer Column: {example['answers']['text'][0]}")

    print("-" * 80)