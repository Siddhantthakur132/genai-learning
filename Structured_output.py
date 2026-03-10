from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict
load_dotenv()

llm=HuggingFaceEndpoint(

  repo_id="meta-llama/Llama-3.1-8B-Instruct",
 task="text-generation"
)

model=ChatHuggingFace(llm=llm)

#schema

class review(TypedDict):
  summary: str
  sentiment:str

st_model=model.with_structured_output(review)

result=st_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")



print(result)