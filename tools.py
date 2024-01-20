import os
from langchain_openai import ChatOpenAI

os.environ['OPENAI_API_KEY'] = 'ojac-vwyiehgkyt'
os.environ['OPENAI_BASE_URL'] = 'https://ojac.jinyuzhineng.com/open/ai/v1'

llm = ChatOpenAI(model='gpt-3.5-turbo')
llm.invoke("how can langsmith help with testing?")