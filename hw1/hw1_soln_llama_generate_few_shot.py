import os
from groq import Groq

client = Groq(
    api_key='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',  # Replace with your actual Groq API key
)

few_shot_examples = """
The following is a set of questions answered in an ELI5 (Explain Like I'm 5) style:

Q: How did climate change and conservation become such a political issue?
A: Unlike other "scientific debates," the impacts of climate change are heavily dispersed and unequal. Moreover, methods to reduce climate change are extraordinarily expensive, affecting major corporations like oil companies. Politicians in areas dependent on these industries may deny climate change to protect jobs and the economy, while other politicians might prioritize different goals.

Q: How did we vividly learn about the lives & history of people or events that happened way back (Like BC era/Socrates' life/stuff like that)?
A: Mostly from writers of that time whose works were preserved. We reconstruct the past from many sources, like tax records, writings, and archeological evidence. Historians gather this information, but they can't always be sure it's completely accurate.

Q: Why not just kill all mosquitoes?
A: We have developed techniques to eradicate certain types of mosquitoes. However, completely killing all mosquitoes could have unforeseen ecological consequences. For now, experiments focus on species that spread diseases like malaria, but wiping them all out might have unintended effects we can't predict.
"""

questions = [
    "Why, in 2017, is Russia considered an enemy of the United States?",
    "Why is the latest Pepsi ad causing so much political outrage?",
    "How are dogs able to return home after many years after running away/getting lost?",
    "Why is a bakers dozen 13 and not 12?",
    "What are these Facebook pages with weird characters and double spacing in the name? They appear to post memes."
]

for question in questions:
    full_prompt = f"{few_shot_examples}\nNow, based on this format, please answer the following question in the ELI5 style:\nQ: {question}"

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": full_prompt,
            }
        ],
        model="llama3-8b-8192", 
    )

    print(f"Question: {question}")
    print(f"Generated Answer: {chat_completion.choices[0].message.content}")
    print("-" * 80)