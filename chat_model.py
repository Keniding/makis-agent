import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

load_dotenv()

# Format: "https://resource_name.ai.azure.com/api/projects/project_name"
PROJECT_ENDPOINT = os.environ["PROJECT_ENDPOINT"]
MODEL = os.environ["MODEL"]

# Create project and openai clients to call Foundry API
project = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(),
)

openai = project.get_openai_client()

# Run a responses API call
response = openai.responses.create(
    model=MODEL,
    input="What is the size of France in square miles?",
)
print(f"Response output: {response.output_text}")
