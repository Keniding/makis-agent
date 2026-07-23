import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, FileSearchTool

load_dotenv()

PROJECT_ENDPOINT = os.environ["PROJECT_ENDPOINT"]
AGENT_NAME = os.environ["AGENT_NAME"]
MODEL = os.environ["MODEL"]

# Carpeta donde pones tus .md con info de makis (uno por maki, o uno con todos)
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

# Guarda vector_store_id, hashes de archivos y si el agente ya existe, para
# no recrear/reindexar/reversionar nada que no haya cambiado.
STATE_PATH = Path(__file__).parent / ".rag_state.json"


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


project = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=DefaultAzureCredential(),
    allow_preview=True,  # necesario para get_openai_client(agent_name=...)
)

# Este cliente no está atado a ningún agente todavía; sirve para operar
# vector stores/files, que son recursos independientes del agente.
openai = project.get_openai_client()

state = load_state()

# 1. Reusar el vector store si ya existe; crearlo solo la primera vez
vector_store = None
if state.get("vector_store_id"):
    try:
        vector_store = openai.vector_stores.retrieve(state["vector_store_id"])
        print(f"Vector store reutilizado (id: {vector_store.id})")
    except Exception:
        vector_store = None  # el store fue borrado o el id ya no es válido

if vector_store is None:
    vector_store = openai.vector_stores.create(name="MakisKnowledgeStore")
    state["vector_store_id"] = vector_store.id
    state["file_hashes"] = {}
    state["file_ids"] = {}
    print(f"Vector store creado (id: {vector_store.id})")

state.setdefault("file_hashes", {})
state.setdefault("file_ids", {})

# 2. Subir solo los .md nuevos o modificados (por hash) al vector store
md_files = sorted(KNOWLEDGE_DIR.glob("*.md"))
if not md_files:
    raise SystemExit(
        f"No encontré archivos .md en {KNOWLEDGE_DIR}. "
        "Crea la carpeta y agrega al menos un .md con info de makis."
    )

reindexed = False
for path in md_files:
    name = path.name
    digest = file_hash(path)
    if state["file_hashes"].get(name) == digest:
        continue  # sin cambios, no reindexar (evita costo de embeddings)

    # Si ya había una versión previa de este archivo, hay que borrarla antes
    # de subir la nueva (los archivos de un vector store son inmutables).
    old_file_id = state["file_ids"].get(name)
    if old_file_id:
        try:
            openai.vector_stores.files.delete(
                vector_store_id=vector_store.id, file_id=old_file_id
            )
        except Exception:
            pass

    with path.open("rb") as f:
        uploaded = openai.vector_stores.files.upload_and_poll(
            vector_store_id=vector_store.id, file=f
        )
    state["file_ids"][name] = uploaded.id
    state["file_hashes"][name] = digest
    reindexed = True
    print(f"  - {name} (re)indexado (file id: {uploaded.id})")

if not reindexed:
    print("Sin cambios en knowledge/ — reutilizando índice existente.")

# 3. Crear el agente con la tool de File Search SOLO la primera vez.
#    Reutilizar el mismo vector_store_id significa que los archivos nuevos
#    quedan buscables sin necesidad de versionar el agente de nuevo.
INSTRUCTIONS = """Eres un experto en makis (sushi rolls) japoneses.

Usa SIEMPRE la herramienta de búsqueda de archivos (file_search) para
responder preguntas sobre ingredientes de makis — tu base de conocimiento
tiene la información autorizada, no inventes ni completes con memoria propia
si el dato no aparece ahí.

Formato de respuesta:
- Nombre del maki
- Ingredientes (arroz, alga nori, relleno, cobertura/topping si aplica)
- Cita brevemente de qué documento sacaste la info si es relevante

Si la pregunta es sobre un maki que no está en la base de conocimiento, dilo
honestamente en vez de inventar ingredientes.

Si te preguntan algo que no tiene que ver con makis o sushi, responde
amablemente que solo puedes ayudar con eso."""

if not state.get("agent_created"):
    agent = project.agents.create_version(
        agent_name=AGENT_NAME,
        definition=PromptAgentDefinition(
            model=MODEL,
            instructions=INSTRUCTIONS,
            tools=[FileSearchTool(vector_store_ids=[vector_store.id])],
        ),
    )
    state["agent_created"] = True
    print(f"Agent creado con RAG (name: {agent.name}, version: {agent.version})\n")
else:
    print(f"Agent '{AGENT_NAME}' ya existía — reutilizando su versión actual.\n")

save_state(state)

# 4. Cliente de chat atado al agente (ya trae instructions + tool + vector store)
chat_client = project.get_openai_client(agent_name=AGENT_NAME)
conversation = chat_client.conversations.create()

if __name__ == "__main__":
    print("Pregúntame por los ingredientes de un maki (Ctrl+C para salir)\n")
    try:
        while True:
            question = input("Tú: ").strip()
            if not question:
                continue
            response = chat_client.responses.create(
                conversation=conversation.id,
                input=question,
            )
            print(f"Agente: {response.output_text}\n")
    except KeyboardInterrupt:
        print("\nHasta luego!")