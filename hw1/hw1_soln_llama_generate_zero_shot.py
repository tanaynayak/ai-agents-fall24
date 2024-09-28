import os
from groq import Groq

client = Groq(
    api_key='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',
)


questions = [
    "Why, in 2017, is Russia considered an enemy of the United States?",
    "Why is the latest Pepsi ad causing so much political outrage?",
    "How are dogs able to return home after many years after running away/getting lost?",
    "Why is a bakers dozen 13 and not 12?",
    "What are these Facebook pages with weird characters and double spacing in the name? They appear to post memes."
]

for question in questions:
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Explain like I'm 5: {question}",
            }
        ],
        model="llama3-8b-8192",
    )

    print(f"Question: {question}")
    print(f"Generated Answer: {chat_completion.choices[0].message.content}")
    print("-" * 80)