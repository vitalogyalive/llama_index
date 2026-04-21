# LlamaIndex — Algorithmes, Bibliothèques & Intégration Gemini

## Vue d'ensemble

**LlamaIndex** est un framework Python open-source pour construire des applications LLM agentiques connectées à des données privées. Il joue le rôle de pont entre les LLMs (GPT-4, Claude, Gemini, Llama, etc.) et vos propres sources de données (PDFs, BDD, APIs, etc.).

### Structure du dépôt (monorepo modulaire)

```
llama_index/
├── llama-index-core/          # Framework central
├── llama-index/               # Package starter (core + OpenAI)
├── llama-index-integrations/  # ~300 plugins
│   ├── llms/        (106+ fournisseurs LLM)
│   ├── embeddings/  (68 fournisseurs)
│   ├── readers/     (160+ formats de données)
│   ├── vector_stores/ (Pinecone, Weaviate, Chroma…)
│   └── tools/, agent/, retrievers/…
├── llama-index-instrumentation/ # Observabilité / télémétrie
└── llama-dev/                 # Outils de développement
```

---

## Algorithmes utilisés

### 1. Similarité vectorielle (`base/embeddings/base.py`)

Quatre modes implémentés via NumPy :

| Mode | Formule | Usage |
|---|---|---|
| **Cosine** (défaut) | `dot(v1,v2) / (‖v1‖ × ‖v2‖)` | Recherche sémantique standard |
| **Dot Product** | `dot(v1, v2)` | Rapide, vecteurs normalisés |
| **Euclidean** | `-‖v1 - v2‖` | Distance géométrique |
| **MMR** | Relevance - λ × similarity_to_selected | Diversité des résultats |

Le top-k utilise un **min-heap** pour sélectionner efficacement les meilleurs résultats.

---

### 2. Pipeline RAG complet

```
Document → Chunking → Embedding → Index → Retrieval → Reranking → LLM → Réponse
```

**Chunking / Splitting :**

| Splitter | Comportement |
|---|---|
| `SentenceSplitter` | 1024 tokens, overlap 200, respecte les phrases (NLTK) |
| `SemanticSplitter` | Coupe sur les ruptures sémantiques |
| `TokenTextSplitter` | Basé sur tiktoken (comptage de tokens exact) |

**Modes de requête vectorielle (`VectorStoreQueryMode`) :**

| Mode | Description |
|---|---|
| `DEFAULT` | Dense (vecteurs uniquement) |
| `SPARSE` | BM25 / sparse |
| `HYBRID` | Dense + sparse avec paramètre `alpha` |
| `SVM` | Support Vector Machine (via scikit-learn) |
| `LinearRegression` | Régression linéaire (via scikit-learn) |
| `LogisticRegression` | Régression logistique (via scikit-learn) |

---

### 3. Reranking (post-processing)

| Algorithme | Mécanisme |
|---|---|
| `LLMRerank` | Demande au LLM de classer N candidats (batch=10) |
| `SentenceTransformerRerank` | Re-embedding sémantique |
| `RankGPTRerank` | Comparaison par paires |
| `SimilarityPostprocessor` | Filtre par seuil de score |
| `RecencyPostprocessor` | Pondération temporelle |

---

### 4. Knowledge Graph

- **KnowledgeGraphIndex** (legacy) : extraction de triplets `(sujet, relation, objet)` via LLM, stockage dans un GraphStore, max 10 triplets/chunk
- **PropertyGraphIndex** (nouveau) : requêtes Text-to-Cypher, synonymes LLM, Vector+Graph hybride — utilise **NetworkX** pour les opérations de graphe

---

## Bibliothèques clés utilisées en interne

| Bibliothèque | Rôle |
|---|---|
| **NumPy** | Calculs vectoriels (dot, norm) |
| **Pydantic v2** | Validation des schémas de données |
| **NLTK** | Tokenisation en phrases |
| **Tiktoken** | Comptage précis des tokens |
| **NetworkX** | Graphes de connaissances |
| **SQLAlchemy** | Requêtes sur données structurées |
| **Tenacity** | Retry avec backoff exponentiel |
| **scikit-learn** | SVM/régression pour retrieval ML |
| **fsspec** | Abstraction filesystem (persistance) |

---

## Utiliser LlamaIndex avec Gemini (Python)

L'intégration existe dans :
- `llama-index-integrations/llms/llama-index-llms-google-genai/`
- `llama-index-integrations/embeddings/llama-index-embeddings-google-genai/`

### Installation

```bash
pip install llama-index-llms-google-genai llama-index-embeddings-google-genai
```

### RAG complet avec Gemini

```python
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# Configuration globale
Settings.llm = GoogleGenAI(
    model="gemini-2.0-flash",          # ou gemini-2.5-pro
    api_key=os.environ["GOOGLE_API_KEY"],
    max_retries=3,                      # backoff exponentiel via tenacity
)
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",
    api_key=os.environ["GOOGLE_API_KEY"],
)

# Ingestion + indexation
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Requête
engine = index.as_query_engine(similarity_top_k=5)
response = engine.query("Explique le concept principal")
print(response)
```

### Agent avec outils (agentic)

```python
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool

def search_web(query: str) -> str:
    """Recherche sur le web."""
    return f"Résultats pour: {query}"

tool = FunctionTool.from_defaults(fn=search_web)
agent = ReActAgent.from_tools([tool], llm=Settings.llm, verbose=True)
response = agent.chat("Cherche les dernières news sur l'IA")
```

### Gemini avec Vertex AI (production)

```python
Settings.llm = GoogleGenAI(
    model="gemini-2.0-flash",
    vertexai_config={
        "project": "mon-projet-gcp",
        "location": "us-central1",
    }
)
```

### Points clés de l'intégration Gemini

- Le LLM supporte le **function calling** natif (tool use)
- Les embeddings ont un retry automatique sur erreurs 429/502-504 (min 1s, max 10s)
- Le modèle par défaut dans le code source est `gemini-3-flash-preview` — à remplacer par `gemini-2.0-flash` ou `gemini-2.5-pro` selon vos besoins
