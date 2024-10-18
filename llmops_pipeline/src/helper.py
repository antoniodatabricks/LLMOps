# Databricks notebook source
import mlflow
import mlflow.deployments
from mlflow import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import WorkspaceObjectAccessControlRequest, WorkspaceObjectPermissionLevel
from databricks.sdk.service.serving import ServingEndpointAccessControlRequest, ServingEndpointPermissionLevel
import requests
import json
import re

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

def get_latest_model_version(model_name_in:str = None):
    """
    Get latest version of registered model
    """
    client = MlflowClient()
    model_version_infos = client.search_model_versions("name = '%s'" % model_name_in)

    if model_version_infos:
      return max([int(model_version_info.version) for model_version_info in model_version_infos])
    else:
      return None

# COMMAND ----------

class Tag:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def as_dict(self):
        return {'key': self.key, 'value': self.value}
    
def deploy_model_serving_endpoint(endpoint_name, endpoint_config, host):
    
    # Initiate the workspace client
    w = WorkspaceClient()
    
    # Get endpoint if it exists
    existing_endpoint = next(
        (e for e in w.serving_endpoints.list() if e.name == endpoint_name), None
    )

    serving_endpoint_url = f"{host}/ml/endpoints/{endpoint_name}"

    # If endpoint doesn't exist, create it
    if existing_endpoint == None:

        # TODO: Tags be parameterized
        tags = [Tag("team", "data science")]
        print(f"Creating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
        w.serving_endpoints.create_and_wait(name=endpoint_name, config=endpoint_config, tags = tags)

        # TODO: Permissions should be parameterized
        print(f"Setting up permissions to the endpoint {endpoint_name}...")
        serving_endpoint_id = w.serving_endpoints.get(endpoint_name).id
        access_control_list=[
            ServingEndpointAccessControlRequest(
                user_name="dummyuser@abc.com",
                permission_level=ServingEndpointPermissionLevel.CAN_VIEW
                )]
        w.serving_endpoints.set_permissions(serving_endpoint_id=serving_endpoint_id, access_control_list=access_control_list)

    # If endpoint does exist, update it to serve the new version
    else:
        print(f"Updating the endpoint {serving_endpoint_url}, this will take a few minutes to package and deploy the endpoint...")
        w.serving_endpoints.update_config_and_wait(served_entities=endpoint_config.served_entities, name=endpoint_name)

# COMMAND ----------

def get_model_optimization_info(model_name, model_version, host, token):

    headers = {"Context-Type": "text/json", "Authorization": f"Bearer {token}"}

    optimizable_info = requests.get(url=f"{host}/api/2.0/serving-endpoints/get-model-optimization-info/{model_name}/{model_version}", headers=headers).json()

    return optimizable_info

# COMMAND ----------

def query_llamaguard(chat, unsafe_categories, llma_guard_endpoint, llma_guard_endpoint_token):
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

def query_llamaguard_sdk(chat, unsafe_categories, llma_guard_endpoint_name):

    client = mlflow.deployments.get_deploy_client("databricks")

    query_payload = {
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
    
    response = client.predict(
            endpoint="llamaguard",
            inputs=query_payload)

    response_list = response["choices"][0]["text"].split("\n")
    result = response_list[0].strip()

    if result == "safe":
        return True, 0
    else:
        category = response_list[1].strip()

    return False, category

# COMMAND ----------

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

# Return the string contents of the most recent messages: [{...}] from the user to be used as input question
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

# Method to format the docs returned by the retriever into the prompt (keep only the text from chunks)
def format_context(docs):
    chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
    return "".join(chunk_contents)

# COMMAND ----------

def test_serving_endpoint(endpoint_name, host, token):
    
    data = {
        "messages": 
            [ 
             {
                 "role": "user", 
                 "content": "What is GenAI?"
             }
            ]
           }
    
    headers = {"Context-Type": "text/json", "Authorization": f"Bearer {token}"}
    
    response = requests.post(
        url=f"{host}/serving-endpoints/{endpoint_name}/invocations", json=data, headers=headers
    )
    
    # Assert that the status code indicates success (2xx range)
    assert response.status_code == 200, f"Model Serving Endpoint: Expected status code 200 but got {response.status_code}"

    # You could also use the requests built-in success check:
    assert response.ok, f"Model Serving Endpoint: Request failed with status code {response.status_code}"

# COMMAND ----------

def score_model(question, host, endpoint, token, break_if_error = False):

    data = {
        "messages": 
            [ 
             {
                 "role": "user", 
                 "content": question
             }
            ]
           }

    headers = {"Context-Type": "text/json", "Authorization": f"Bearer {token}"}

    response = requests.post(
        url=f"{host}/serving-endpoints/{endpoint}/invocations", json=data, headers=headers
    )

    if response.status_code != 200:
        if break_if_error:
            raise Exception(f"Model Serving Endpoint {endpoint}: Expected status code 200 but got {response.status_code}")
        else:
            return ""
    else:
        return response.text
