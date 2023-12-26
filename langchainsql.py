import os

from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor

os.environ['OPENAI_API_KEY'] = 'sk-cW09uf6xHLQevycw4ZPMwOWb7f7vloBcuPx68z8FPNfnENMF'
os.environ['OPENAI_API_BASE'] = 'https://api.zchat.tech/v1'

llm = OpenAI(temperature=0)
tools = load_tools(['llm-math'], llm=llm)

db_user = "root"
db_password = "123456"
db_host = "10.82.77.104"
db_name = "nl2sql"
db = SQLDatabase.from_uri("mysql+pymysql://root:123456@10.82.77.104/nl2sql")

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

#agent_executor.run("描述与订单相关的表及其关系")
agent_executor.run("张三今年有哪些订单")