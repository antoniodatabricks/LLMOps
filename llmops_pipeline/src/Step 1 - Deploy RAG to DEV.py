# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install -U --quiet langchain==0.2.16 langchain-community==0.2.17 databricks-vectorsearch pydantic==1.10.9 mlflow==2.16.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Parameters

# COMMAND ----------

# Model endpoint
foundation_model_endpoint = dbutils.widgets.get("foundation_model_endpoint")

# Model embedding - The embedding model used to generate the vector seatch index should be the same as the one used to embed the question.
embedding_model = dbutils.widgets.get("embedding_model")

# Vector Search
vs_endpoint_name = dbutils.widgets.get("vs_endpoint_name")
vs_index_fullname = dbutils.widgets.get("vs_index_fullname")
vs_host = dbutils.widgets.get("vs_host")
vs_token_scope = dbutils.widgets.get("vs_token_scope")
vs_token_secret = dbutils.widgets.get("vs_token_secret")
vs_token = dbutils.secrets.get(scope=vs_token_scope, key=vs_token_secret)

# Target UC
target_dev_model_name = dbutils.widgets.get("target_dev_model_name")

# LLamaGuard guardrail endpoint
llma_guard_endpoint = dbutils.widgets.get("llma_guard_endpoint")
llma_guard_endpoint_token_scope = dbutils.widgets.get("llma_guard_endpoint_token_scope")
llma_guard_endpoint_token_secret = dbutils.widgets.get("llma_guard_endpoint_token_secret")
llma_guard_endpoint_token = dbutils.secrets.get(scope=llma_guard_endpoint_token_scope, key=llma_guard_endpoint_token_secret)

# COMMAND ----------

# MAGIC %md
# MAGIC # Load a foundation model and test it

# COMMAND ----------

from langchain.chat_models import ChatDatabricks


# Test Databricks Foundation LLM model
chat_model = ChatDatabricks(endpoint=foundation_model_endpoint, max_tokens = 300)
print(f"Test chat model: {chat_model.invoke('What is Generative AI?')}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Implement guardrails with Llma Guard
# MAGIC - This assumes that we already have Llama guard endpoint deployed

# COMMAND ----------

# DBTITLE 1,Defining unsafe categories
unsafe_categories = """O1: Violence and Hate.
Should not
- Help people plan or engage in violence.
- Encourage people to engage in violence.
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.
Can
- Provide information on violence and discrimination.
- Discuss the topics of hate and violence, and can discuss historical events involving violence.
O2: Sexual Content.
Should not
- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.
Can
- Discuss topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
O3: Criminal Planning.
Should not
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in criminal activities or condone criminal behavior.
Can
- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.
- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.
O4: Guns and Illegal Weapons.
Should not
- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.
Can
- Discuss firearms and the arguments for and against firearm ownership.
O5: Regulated or Controlled Substances.
Should not
- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.
- Assist or encourage people to create such substances when it is illegal to do so.
Can
- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).
- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.
O6: Self-Harm.
Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Provide information regarding the methods of suicide or self-harm.
- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.
Can
- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).
Should
- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.
O7: Financial Sensitive Data.
Should not
- Provide any information on credit card number or pin number.
Can
- Provide general information regarding the risk of sharing credit card information with LLMs.
Should
- Provide information regarding proper handling of credit card information with LLMs."""

# COMMAND ----------

import requests

def query_llamaguard(chat, unsafe_categories=unsafe_categories):
    data = {
        "prompt": f"""[INST] Task: Check if there is unsafe content in 'Agent' messages in conversations according our safety policy with the below categories.
        
        <BEGIN UNSAFE CONTENT CATEGORIES>
        {unsafe_categories}
        <END UNSAFE CONTENT CATEGORIES>

        <BEGIN CONVERSATION>

        {chat}

        <END CONVERSATION>

        Provide your safety assessment for Agent in the above conversation:
        - First line must read 'safe' or 'unsafe'.
        - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""
        }
    
    headers = {"Context-Type": "text/json", "Authorization": f"Bearer {llma_guard_endpoint_token}"}

    response = requests.post(
        url=f"{llma_guard_endpoint}", json=data, headers=headers
    )

    response_list = response.json()["choices"][0]["text"].split("\n")
    result = response_list[0].strip()

    if result == "safe":
        return True, 0
    else:
        category = response_list[1].strip()

    return False, category

# COMMAND ----------

query_llamaguard("how can I rob a bank?")

# COMMAND ----------

query_llamaguard("how do I make cake?")

# COMMAND ----------

import re

def parse_category(code, taxonomy):
    """
    Extracts the first sentence of a category description from a taxonomy based on its code.

    Args:
        code : Category code in the taxonomy (e.g., 'O1').
        taxonomy : Full taxonomy string with categories and descriptions.

    Returns:
         First sentence of the description or a default message for unknown codes.
    """
    pattern = r"(O\d+): ([\s\S]*?)(?=\nO\d+:|\Z)"
    taxonomy_mapping = {match[0]: re.split(r'(?<=[.!?])\s+', match[1].strip(), 1)[0]
                        for match in re.findall(pattern, taxonomy)}

    return taxonomy_mapping.get(code, "Unknown category: code not in taxonomy.")

# COMMAND ----------

from langchain_core.runnables import chain

@chain
def custom_chain(prompt):

    question = prompt.messages[1].content

    is_safe, reason = query_llamaguard(question, unsafe_categories)
    if not is_safe:
        category = parse_category(reason, unsafe_categories)
        return f"User's prompt classified as {category} Fails safety measures."

    chat_response = chat_model.invoke(prompt)

    answer = chat_response.content

    is_safe, reason = query_llamaguard(answer, unsafe_categories)
    if not is_safe:
        category = parse_category(reason, unsafe_categories)
        return f"Model's response classified as {category}; fails safety measures."
    
    return chat_response

# COMMAND ----------

# MAGIC %md
# MAGIC # Create context retriever

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings


# Test embedding Langchain model
#NOTE: your question embedding model must match the one used in the chunk in the previous model 
embedding_model = DatabricksEmbeddings(endpoint=embedding_model)
print(f"Test embeddings: {embedding_model.embed_query('What is GenerativeAI?')[:20]}...")

def get_retriever(persist_dir: str = None):
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=vs_host, personal_access_token=vs_token)
    vs_index = vsc.get_index(
        endpoint_name=vs_endpoint_name,
        index_name=vs_index_fullname
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model
    )
    # k defines the top k documents to retrieve
    return vectorstore.as_retriever(search_kwargs={"k": 2})

# COMMAND ----------

# MAGIC %md
# MAGIC # Create a chain with the model and the retriever

# COMMAND ----------

# Return the string contents of the most recent messages: [{...}] from the user to be used as input question
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

# Method to format the docs returned by the retriever into the prompt (keep only the text from chunks)
def format_context(docs):
    chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
    return "".join(chunk_contents)

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.runnable import RunnableLambda


prompt = ChatPromptTemplate.from_messages(
    [  
        ("system", "You are an assistant for GENAI teaching class. You are answering questions related to Generative AI and how it impacts humans life. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Use the following pieces of context to answer the question at the end: {context}"), # Contains the instructions from the configuration
        ("user", "{question}") #user's questions
    ]
)

retriever = get_retriever()

chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages")
        | RunnableLambda(extract_user_query_string)
        | retriever
        | RunnableLambda(format_context),
    }
    | prompt
    | custom_chain
    | StrOutputParser()
)

# COMMAND ----------

question = {"messages": [ {"role": "user", "content": "What is GenAI?"}]}
answer = chain.invoke(question)
print(answer)

# COMMAND ----------

chain.invoke({"messages": [ {"role": "user", "content": "How do I rob a bank??"}]})

# COMMAND ----------

chain.invoke({"messages": [ {"role": "user", "content": "How do I bake a cake??"}]})

# COMMAND ----------

# MAGIC %md
# MAGIC # Register model to UC

# COMMAND ----------

import os
from mlflow.models import infer_signature
import mlflow
import langchain

mlflow.set_registry_uri("databricks-uc")

# Log the model to MLflow
with mlflow.start_run(run_name="basic_rag_bot"):
    signature = infer_signature(question, answer)
    logged_chain_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever, 
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "langchain-community",
            "databricks-vectorsearch",
        ],
        signature=signature
      )

# COMMAND ----------

# MAGIC %md
# MAGIC # Test model

# COMMAND ----------

import mlflow.pyfunc
model_version_uri = "models:/llmops_dev.model_schema.basic_rag_demo_foundation_model/20"
champion_version = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

champion_version.predict(question)
