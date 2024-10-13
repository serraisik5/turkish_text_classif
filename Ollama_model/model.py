from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
import re

def get_classification_chain(labels):
    template = """
    {text} """

    prompt = PromptTemplate(
        input_variables=["text", "labels"],
        template=template,
    )

    model = Ollama(model="lamaturk")
    chain = LLMChain(prompt=prompt, llm=model)
    return chain

def classify_text(chain, text, labels):
    print(f"Classifying text: {text}")  
    response = chain.run({"text": text, "labels": labels})

    if "Cevap:" in response:
        answer = response.split("Cevap:")[-1].strip()
    else:
        answer = response.strip()

    answer = re.sub(r'[^\w\s]', '', answer)
    first_word = answer.split()[0].lower() if answer else ""

    print(f"Predicted category: {first_word}") 
    return answer.lower()
