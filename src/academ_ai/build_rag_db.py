import re
import os
import sqlite3
import weaviate
import weaviate.classes as wvc
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.classes.query import MetadataQuery, QueryReference
from weaviate.classes.config import ReferenceProperty
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dataclasses import dataclass
import logging
from typing import Any
import nltk

# Constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "localhost")
WEAVIATE_PORT = os.environ.get("WEAVIATE_PORT", 8080)
GRPC_PORT = os.environ.get("GRPC_PORT", 50051)

DEFAULT_QUERY_KWARGS = {"alpha": 0.3}
DEFAULT_BATCH_SIZE = 100
CHUNK_OVERLAP = 2

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    logger.info("Checking NLTK data")
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    logger.info("NLTK data not found, downloading")
    nltk.download("punkt")
    nltk.download("punkt_tab")

@dataclass
class PaperChunk:
    """
    Represents a chunk of text from a paper's abstract.
    """

    paper_id: int
    chunk_id: int
    text: str
    title: str
    authors: list[str]
    category: str


class RAGDatabase:
    """
    Builds a RAG database from SQLite papers database.
    """

    def __init__(
        self,
        db_path: str = None,
        model_name: str = EMBEDDING_MODEL_NAME,
        weaviate_url: str = WEAVIATE_URL,
        weaviate_port: int = WEAVIATE_PORT,
    ):
        """
        Initialize the RAG database builder.

        Args:
            db_path (str, optional): Path to the SQLite database. Defaults to
                None.
            model_name (str, optional): Name of the HuggingFace model to use for
                embeddings. Defaults to EMBEDDING_MODEL_NAME.
            weaviate_url (str, optional): URL of the Weaviate instance. Defaults
                to WEAVIATE_URL.
            weaviate_port (int, optional): Port of the Weaviate instance. Defaults
                to WEAVIATE_PORT.
        """
        self.db_path = db_path
        self.model_name = model_name
        self.weaviate_url = weaviate_url
        self.weaviate_port = weaviate_port

        self.embedding_model = None
        self.weaviate_client = None

    def connect_to_database(self):
        """
        Connect to the SQLite database.
        """
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Access columns by name
            logger.info(f"Connected to database at {self.db_path}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            return False

    def get_papers_with_authors(self) -> list[dict[str, Any]]:
        """
        Retrieve all papers with their authors from the database.

        Returns:
            List of dictionaries containing paper and author information
        """
        query = """
        SELECT p.id, p.title, p.abstract, p.category, 
               GROUP_CONCAT(a.first_name || ' ' || a.last_name) as authors
        FROM papers p
        LEFT JOIN authors a ON p.id = a.paper_id
        GROUP BY p.id
        """

        cursor = self.conn.cursor()
        cursor.execute(query)
        papers = []
        for row in cursor.fetchall():
            paper = dict(row)
            paper["authors"] = (
                paper["authors"].split(",") if paper["authors"] else []
            )
            papers.append(paper)

        logger.info(f"Retrieved {len(papers)} papers from database")
        return papers

    def chunk_abstract(
        self,
        abstract: str,
        min_sentence_length: int = 10,
        min_chunk_length: int = 30,
    ) -> list[str]:
        """
        Split abstract into chunks with overlapping sentences.

        Args:
            abstract: The abstract text to chunk
            min_sentence_length: Minimum number of words per sentence to form a chunk
            min_chunk_length: Minimum number of words per chunk

        Returns:
            List of text chunks
        """
        # Split into sentences
        sentences = nltk.sent_tokenize(abstract)
        chunks = []
        current_chunk = []
        current_length = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            words = sentence.split()

            # If sentence meets minimum length requirement
            if len(words) >= min_sentence_length:
                # If we have a chunk in progress, save it and start a new one with overlap
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Start new chunk with overlap
                    overlap = (
                        current_chunk[-CHUNK_OVERLAP:]
                        if len(current_chunk) > CHUNK_OVERLAP
                        else current_chunk
                    )
                    current_chunk = overlap.copy()
                    current_length = sum(len(s.split()) for s in current_chunk)

                # Add current sentence to new chunk
                current_chunk.append(sentence)
                current_length += len(words)
                i += 1
            else:
                # Sentence is too short, add to current chunk
                current_chunk.append(sentence)
                current_length += len(words)
                i += 1

                # If we've reached the minimum chunk length, save it
                if (
                    current_length >= min_chunk_length
                    and len(current_chunk) > 1
                ):
                    chunks.append(" ".join(current_chunk))
                    # Start new chunk with overlap
                    overlap = (
                        current_chunk[-CHUNK_OVERLAP:]
                        if len(current_chunk) > CHUNK_OVERLAP
                        else current_chunk
                    )
                    current_chunk = overlap.copy()
                    current_length = sum(len(s.split()) for s in current_chunk)

        # Add the last chunk if it's not empty
        if current_chunk and (current_length >= min_chunk_length or not chunks):
            chunks.append(" ".join(current_chunk))

        return chunks

    def create_paper_chunks(
        self, papers: list[dict[str, Any]]
    ) -> list[PaperChunk]:
        """
        Create chunks from paper abstracts.

        Args:
            papers: List of paper dictionaries with abstracts

        Returns:
            List of PaperChunk objects
        """
        paper_chunks = []

        for paper in tqdm(papers, desc="Chunking abstracts"):
            abstract = paper.get("abstract", "")
            if not abstract:
                continue

            chunks = self.chunk_abstract(abstract)

            for i, chunk_text in enumerate(chunks):
                paper_chunk = PaperChunk(
                    paper_id=paper["id"],
                    chunk_id=i,
                    text=chunk_text,
                    title=paper["title"],
                    authors=paper["authors"],
                    category=paper["category"],
                )
                paper_chunks.append(paper_chunk)

        logger.info(
            f"Created {len(paper_chunks)} chunks from {len(papers)} papers"
        )
        return paper_chunks

    def connect_to_weaviate(self) -> bool:
        """
        Connect to Weaviate instance.

        Returns:
            bool: True if connection was successful, False otherwise
        """
        logger.info(
            f"Connecting to Weaviate at {self.weaviate_url}:{self.weaviate_port}"
        )
        self.weaviate_client = weaviate.connect_to_local(
            host=self.weaviate_url,
            port=self.weaviate_port,
            grpc_port=GRPC_PORT,
            additional_config=AdditionalConfig(
                timeout=Timeout(init=2, query=45, insert=120)
            ),
        )
        logger.info(
            f"Connected to Weaviate at {self.weaviate_url}:{self.weaviate_port}"
        )
        return True

    def get_schema_definition(self) -> dict:
        """
        Get the schema definition for the PaperChunk class.

        Returns:
            dict: Schema definition
        """
        return {
            "class": "PaperChunk",
            "description": "A chunk of text from a paper's abstract with metadata",
            "properties": [
                wvc.config.Property(
                    name="paper_id",
                    data_type=wvc.config.DataType.INT,
                    description="ID of the paper in the SQLite database",
                ),
                wvc.config.Property(
                    name="chunk_id",
                    data_type=wvc.config.DataType.INT,
                    description="ID of the chunk within the paper",
                ),
                wvc.config.Property(
                    name="text",
                    data_type=wvc.config.DataType.TEXT,
                    description="The chunk of text from the abstract",
                ),
                wvc.config.Property(
                    name="title",
                    data_type=wvc.config.DataType.TEXT,
                    description="Title of the paper",
                ),
            ],
            "references": [
                ReferenceProperty(
                    name="information",
                    target_collection="PaperAbstract",
                    description="Paper information for chunks",
                ),
            ],
            "vectorizer": "none",
        }

    def get_abstract_schema_definition(self) -> dict:
        """
        Get the schema definition for the PaperAbstract class.

        Returns:
            dict: Schema definition
        """
        return {
            "class": "PaperAbstract",
            "description": "Abstract of a paper with metadata",
            "properties": [
                wvc.config.Property(
                    name="paper_id",
                    data_type=wvc.config.DataType.INT,
                    description="ID of the paper in the SQLite database",
                ),
                wvc.config.Property(
                    name="title",
                    data_type=wvc.config.DataType.TEXT,
                    description="Title of the paper",
                ),
                wvc.config.Property(
                    name="authors",
                    data_type=wvc.config.DataType.TEXT_ARRAY,
                    description="List of authors of the paper",
                ),
                wvc.config.Property(
                    name="category",
                    data_type=wvc.config.DataType.TEXT,
                    description="Category of the paper",
                ),
                wvc.config.Property(
                    name="abstract",
                    data_type=wvc.config.DataType.TEXT,
                    description="Abstract of the paper",
                ),
            ],
            "vectorizer": "none",
        }

    def initialize_schema(self) -> bool:
        """
        Initialize the Weaviate schema if it doesn't exist.

        Returns:
            bool: True if schema was created or already exists, False on error
        """
        if not hasattr(self, "weaviate_client") or not self.weaviate_client:
            if not self.connect_to_weaviate():
                return False

        class_name = "PaperChunk"
        if self.weaviate_client.collections.exists(class_name):
            logger.warning(
                f"Collection '{class_name}' already exists. Using existing collection."
            )
            return True

        try:
            schema_definition = self.get_schema_definition()
            self.weaviate_client.collections.create(
                schema_definition["class"],
                description=schema_definition["description"],
                properties=schema_definition["properties"],
                references=schema_definition["references"],
            )
            logger.info(f"Created chunks collection for '{class_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to create chunks collection: {e}")
            return False

    def initialize_abstract_schema(self) -> bool:
        """
        Initialize the Weaviate abstract schema if it doesn't exist.

        Returns:
            bool: True if schema was created or already exists, False on error
        """
        if not hasattr(self, "weaviate_client") or not self.weaviate_client:
            if not self.connect_to_weaviate():
                return False

        class_name = "PaperAbstract"
        if self.weaviate_client.collections.exists(class_name):
            logger.warning(
                f"Collection '{class_name}' already exists. Using existing collection."
            )
            return True

        try:
            schema_definition = self.get_abstract_schema_definition()
            self.weaviate_client.collections.create(
                schema_definition["class"],
                description=schema_definition["description"],
                properties=schema_definition["properties"],
            )
            logger.info(f"Created collection for '{class_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False

    def delete_schema(self, confirm: bool = False) -> bool:
        """
        Delete the Weaviate schema if it exists.

        Args:
            confirm: Must be set to True to actually delete the schema

        Returns:
            bool: True if schema was deleted or didn't exist, False on error
        """
        if not confirm:
            logger.warning(
                "Schema deletion not confirmed. Set confirm=True to delete."
            )
            return False

        if not hasattr(self, "weaviate_client") or not self.weaviate_client:
            if not self.connect_to_weaviate():
                return False

        error, exception = False, None
        for class_name in ["PaperChunk", "PaperAbstract"]:
            if not self.weaviate_client.collections.exists(class_name):
                logger.warning(
                    f"Collection '{class_name}' does not exist. Nothing to delete."
                )
                continue

            try:
                self.weaviate_client.collections.delete(class_name)
                logger.info(f"Deleted collection '{class_name}'")
            except Exception as e:
                logger.error(f"Failed to delete collection: {e}")
                error = True
                exception = e

        if error:
            logger.error(f"Failed to delete schema: {exception}")
            return False

        return True

    def initialize_embedding_model(self):
        """
        Initialize the embedding model.
        """
        logger.info(f"Loading embedding model: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        if not self.embedding_model:
            self.initialize_embedding_model()

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def get_existing_paper_ids(self) -> set[int]:
        """
        Get set of paper IDs that are already indexed in Weaviate.

        Returns:
            set[int]: Set of paper IDs that are already indexed
        """
        if not hasattr(self, "weaviate_client") or not self.weaviate_client:
            return set()

        try:
            all_paper_ids = set(
                [
                    item.properties["paper_id"]
                    for item in self.weaviate_client.collections.get(
                        "PaperChunk"
                    ).iterator()
                ]
            )
            return all_paper_ids
        except Exception as e:
            logger.error(f"Error getting existing paper IDs: {e}")
            return set()

    def index_paper_chunks(
        self,
        paper_chunks: list[PaperChunk],
        uuid_correspondence: dict[int, str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        existing_paper_ids: set[int] | None = None,
    ):
        """
        Index paper chunks in Weaviate with embeddings.

        Args:
            paper_chunks: List of PaperChunk objects to index.
            uuid_correspondence: Dictionary mapping paper IDs to UUIDs.
            batch_size: Batch size for indexing.
            existing_paper_ids: Set of paper IDs that are already indexed.
        """
        if not hasattr(self, "weaviate_client") or not self.weaviate_client:
            if not self.connect_to_weaviate() or not self.initialize_schema():
                raise RuntimeError(
                    "Failed to connect to Weaviate or initialize schema"
                )

        # Filter out already indexed papers if in incremental mode
        if existing_paper_ids:
            original_count = len(paper_chunks)
            paper_chunks = [
                chunk
                for chunk in paper_chunks
                if chunk.paper_id not in existing_paper_ids
            ]
            logger.info(
                f"Skipping {original_count - len(paper_chunks)} already indexed papers"
            )

            if not paper_chunks:
                logger.info("All papers already indexed")
                return

        # Process in batches
        for i in tqdm(
            range(0, len(paper_chunks), batch_size),
            desc="Indexing chunks",
            unit="batch",
        ):
            batch = paper_chunks[i : i + batch_size]

            # Prepare batch data
            texts = [chunk.text for chunk in batch]
            embeddings = self.embed_texts(texts)

            # Prepare objects for batch import
            objects = []
            for chunk, vector in zip(batch, embeddings):
                obj = {
                    "paper_id": chunk.paper_id,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "title": chunk.title,
                    "authors": chunk.authors,
                    "category": chunk.category,
                    "vector": vector,
                }
                objects.append(obj)

            # Import batch to Weaviate
            try:
                with self.weaviate_client.collections.get(
                    "PaperChunk"
                ).batch.fixed_size(batch_size=batch_size) as batch:
                    for obj in objects:
                        batch.add_object(
                            properties={
                                k: v for k, v in obj.items() if k != "vector"
                            },
                            references={
                                "information": uuid_correspondence[
                                    obj["paper_id"]
                                ]
                            },
                            vector=obj["vector"],
                        )
            except Exception as e:
                logger.error(f"Error importing batch {i//batch_size}: {e}")
                raise

        logger.info(f"Indexed {len(paper_chunks)} paper chunks in Weaviate")

    def build(
        self, incremental: bool = True, batch_size: int = DEFAULT_BATCH_SIZE
    ):
        """
        Build or update the RAG database.

        Args:
            incremental: If True, only index new papers that aren't already indexed
            batch_size: Batch size for indexing

        Returns:
            bool: True if the operation was successful, False otherwise
        """
        if self.db_path is None:
            logger.error("Database path (db_path) not provided")
            return False

        # Connect to databases
        if not self.connect_to_database():
            return False

        if not self.connect_to_weaviate():
            return False

        if not self.initialize_abstract_schema():
            return False

        if not self.initialize_schema():
            return False

        if incremental:
            existing_paper_ids = self.get_existing_paper_ids()
        else:
            existing_paper_ids = None

        try:
            # Initialize embedding model
            self.initialize_embedding_model()

            # Get papers with authors
            papers = self.get_papers_with_authors()
            if not papers:
                logger.error("No papers found in the database")
                return False

            # Create paper chunks
            paper_chunks = self.create_paper_chunks(papers)
            if not paper_chunks:
                logger.error("No paper chunks were created")
                return False

            abstracts = self.weaviate_client.collections.get("PaperAbstract")
            with abstracts.batch.fixed_size(batch_size=batch_size) as batch:
                for paper in papers:
                    if paper["id"] in existing_paper_ids:
                        continue
                    batch.add_object(
                        properties={
                            "paper_id": paper["id"],
                            "title": paper["title"],
                            "abstract": paper["abstract"],
                            "category": paper["category"],
                            "authors": paper["authors"],
                        }
                    )
            uuid_correspondence = {
                o.properties["paper_id"]: o.uuid for o in abstracts.iterator()
            }

            logger.info(
                f"Transferred {len(uuid_correspondence)} abstracts to Weaviate"
            )

            # Index chunks
            self.index_paper_chunks(
                paper_chunks,
                uuid_correspondence=uuid_correspondence,
                batch_size=batch_size,
                existing_paper_ids=existing_paper_ids,
            )

            # Get stats
            if hasattr(self, "weaviate_client"):
                try:
                    result = self.weaviate_client.collections.get(
                        "PaperChunk"
                    ).aggregate.over_all(total_count=True)

                    logger.info(
                        f"Total chunks in database: {result.total_count}"
                    )
                except Exception as e:
                    logger.warning(f"Could not get chunk count: {e}")

            logger.info("RAG database build completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error building RAG database: {e}", exc_info=True)
            return False
        finally:
            # Close database connection
            if hasattr(self, "conn"):
                self.conn.close()
                logger.info("Database connection closed")

    def query_text(
        self,
        text: str,
        limit: int = 10,
        query_kwargs: dict[str, Any] = None,
        group_by_paper: bool = False,
    ) -> list[PaperChunk]:
        """
        Query the RAG database for papers similar to the given text.

        Args:
            text (str): The text to query.
            limit (int, optional): The maximum number of results to return.
                Defaults to 10.
            query_kwargs (dict[str, Any], optional): Additional keyword arguments
                to pass to the query. Defaults to None.
            group_by_paper (bool, optional): If True, groups the outputs by paper.
                Defaults to False.

        Returns:
            List of PaperChunk objects similar to the given text
        """
        # Initialize embedding model
        self.connect_to_weaviate()
        self.initialize_embedding_model()

        if query_kwargs is None:
            query_kwargs = {}
            query_kwargs.update(DEFAULT_QUERY_KWARGS)

        chunks = self.weaviate_client.collections.get("PaperChunk")
        result = chunks.query.hybrid(
            query=text,
            query_properties=["text", "title"],
            vector=self.embed_texts([text])[0],
            limit=limit,
            return_metadata=MetadataQuery(score=True, explain_score=True),
            return_references=QueryReference(link_on="information"),
            **query_kwargs,
        )

        if group_by_paper:
            grouped_results = {}
            for object in result.objects:
                paper_id = object.properties["paper_id"]
                if paper_id not in grouped_results:
                    abstract_obj = object.references["information"].objects[0]
                    abstract = abstract_obj.properties["abstract"]
                    grouped_results[paper_id] = {
                        "abstract": abstract,
                        "title": object.properties["title"],
                        "authors": abstract_obj.properties["authors"],
                        "category": abstract_obj.properties["category"],
                        "retrieved_chunks": [],
                    }
                text_string = re.escape(object.properties["text"].strip())
                search_obj = re.search(text_string, abstract)
                span = search_obj.span() if search_obj else None
                grouped_results[paper_id]["retrieved_chunks"].append(
                    {
                        "text": object.properties["text"],
                        "span": span,
                    }
                )

            return grouped_results

        return result

    def __del__(self):
        if hasattr(self, "weaviate_client"):
            self.weaviate_client.close()
            logger.info("Weaviate connection closed")


def main():
    """
    Main function to manage the RAG database.
    """
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Manage the RAG database for academic papers"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        help="Path to the RAG database SQLite database",
        required=True,
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing schema and rebuild from scratch (warning: this will delete all data)",
    )
    parser.add_argument(
        "--no-incremental",
        action="store_false",
        dest="incremental",
        default=True,
        help="Disable incremental indexing (re-index all papers)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=EMBEDDING_MODEL_NAME,
        help="HuggingFace model to use for embeddings",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for indexing",
    )

    args = parser.parse_args()

    # Initialize the RAG database builder
    rag_builder = RAGDatabase(model_name=args.model, db_path=args.db_path)

    # Handle schema reset if requested
    if args.reset:
        if rag_builder.delete_schema(confirm=True):
            logger.info("✓ Schema deleted successfully")
        else:
            logger.error("Failed to delete schema")
            return

    # Build/update the database
    logger.info("Building RAG database...")
    success = rag_builder.build(
        incremental=args.incremental,
        batch_size=args.batch_size,
    )

    if success:
        logger.info("RAG database updated successfully!")
        logger.info(f"Weaviate instance running at: {rag_builder.weaviate_url}")
    else:
        logger.error(
            "Failed to update RAG database. Check the logs for details."
        )


if __name__ == "__main__":
    main()
