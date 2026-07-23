import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

load_dotenv()

# Format: "https://resource_name.services.ai.azure.com/api/projects/project_name"
PROJECT_ENDPOINT = os.environ["PROJECT_ENDPOINT"]
AGENT_NAME = os.environ["AGENT_NAME"]
MODEL = os.environ["MODEL"]

# allow_preview=True es necesario para poder llamar al agente por agent_name
# (endpoint dedicado del agente, no solo el modelo genérico).
project = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(),
    allow_preview=True,
)

INSTRUCTIONS = """Eres un experto en makis (sushi rolls) japoneses.

Cuando te pregunten por un tipo de maki, responde con sus ingredientes
típicos/genéricos basándote en tu conocimiento general del plato — no
inventes recetas exóticas ni recetas específicas de un restaurante.

Formato de respuesta:
- Nombre del maki
- Ingredientes (arroz, alga nori, relleno, cobertura/topping si aplica)
- Si el maki tiene variantes muy conocidas (ej. California roll clásico
  vs. con mango), acláralo brevemente en una línea.

Si te preguntan por un maki que no reconoces como plato estándar, dilo
honestamente en vez de inventar ingredientes.

Si te preguntan algo que no tiene que ver con makis o sushi, responde
amablemente que solo puedes ayudar con eso."""

# Crea (o versiona) el agente persistente con estas instrucciones
agent = project.agents.create_version(
    agent_name=AGENT_NAME,
    definition=PromptAgentDefinition(
        model=MODEL,
        instructions=INSTRUCTIONS,
    ),
)
print(f"Agent listo (name: {agent.name}, version: {agent.version})\n")

# Cliente OpenAI apuntado al endpoint del agente -> hereda sus instructions
openai = project.get_openai_client(agent_name=AGENT_NAME)

conversation = openai.conversations.create()

if __name__ == "__main__":
    print("Pregúntame por los ingredientes de un maki (Ctrl+C para salir)\n")
    try:
        while True:
            question = input("Tú: ").strip()
            if not question:
                continue
            response = openai.responses.create(
                conversation=conversation.id,
                input=question,
            )
            print(f"Agente: {response.output_text}\n")
    except KeyboardInterrupt:
        print("\nHasta luego!")