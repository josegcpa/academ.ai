# academ.ai 

academ.ai is a tool to retrieve academic papers from biorxiv and medrxiv (or other sources) and store them in a vector database for semantic search. It features workflows for retrieving abstracts from bioRxiv and medRxiv, building a vector database using HuggingFace's SentenceTransformers and Weaviate, and a UI to facilitate the search process.

If you want more information on this I wrote a [small blog post](https://josegcpa.net/blog/2025/academai/) on the topic.

## Installation

To run the entire workflow Docker Compose is necessary. Start by cloning the repository (`git clone https://github.com/josegcpa/academ.ai.git`) and creating a .env file with the following variables:

```bash
WEAVIATE_URL=weaviate
WEAVIATE_PORT=8080
SERVE_PORT=1234
```

Afterwards, run `docker compose up` to start the entire workflow. This will use a sample database in `data/papers_test.db` (with a sample of ~100 papers), but you can build your own with `uv run academ_ai_retrieve`. If you change this remember to update the database path (`/app/data/papers_test.db`) in the `build_db` service in the `docker-compose.yaml` file.