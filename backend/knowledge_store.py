"""
TiDB Database Utilities for Knexion

This module provides database utilities for managing both vector embeddings and knowledge graphs
in TiDB. It handles:
- Vector storage and retrieval for semantic search
- Knowledge graph creation, storage, and querying
- Graph visualization
- Knowledge graph extraction from text using LLM

The module uses TiDB's vector capabilities for semantic search and traditional tables
for storing knowledge graph entities and relationships.
"""

import os
import time
import uuid
from typing import Optional, List, Dict, Any, Callable, Protocol
from collections import deque

# TiDB and database imports
from pytidb import TiDBClient
from pytidb.schema import TableModel, Field, Relationship as SQLRelationship
from pytidb.datatype import JSON, TEXT
from sqlalchemy import select, or_, text
from sqlalchemy.orm import joinedload
from pytidb.embeddings import EmbeddingFunction

# LangChain imports
from langchain_core.documents import Document
from langchain_community.vectorstores import TiDBVectorStore

# LLM and data models
import dspy
from pydantic import BaseModel, Field as PyField

# Visualization
from pyvis.network import Network

# Local imports
from llm_chains import embed_model

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

EMBED_TABLE_NAME = "semantic_embedding"
tidb_connection_string = os.getenv("TIDB_CONNECTION_STRING")

# TiDB client setup
db = TiDBClient.connect(
    host=os.getenv('TIDB_HOST_NAME'),
    port=int(os.getenv("TIDB_PORT_NUMBER")),
    username=os.getenv("TIDB_USERNAME"),
    password=os.getenv("TIDB_PASSWORD"),
    database=os.getenv("TIDB_DATABASE_NAME"),
    ensure_db=True
)

# Embedding function for knowledge graph
openai_embed = EmbeddingFunction(
    model_name="openai/text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

# =============================================================================
# VECTOR STORAGE OPERATIONS
# =============================================================================

def create_vectorstore(thread_id: str):
    """
    Create a TiDB vector store retriever for a specific conversation thread.
    
    Args:
        thread_id: Unique identifier for the conversation thread
    
    Returns:
        Configured retriever that filters results by thread_id
    """
    vector_store = TiDBVectorStore(
        connection_string=tidb_connection_string,
        embedding_function=embed_model,
        table_name=EMBED_TABLE_NAME,
        distance_strategy="cosine"
    )
    
    return vector_store.as_retriever(
        search_kwargs={"k": 10, "filter": {"thread_id": thread_id}},
    )


def ingest_documents(docs: List[Document], meta: Dict[str, Any]) -> None:
    """
    Store documents in the vector database with metadata.
    
    Args:
        docs: List of LangChain Document objects to store
        meta: Metadata dictionary (must include thread_id)
    """
    # Add metadata to all documents
    docs_with_meta = [
        Document(page_content=doc.page_content, metadata=meta) 
        for doc in docs
    ]
    
    # Store in TiDB vector store
    TiDBVectorStore.from_documents(
        documents=docs_with_meta,
        embedding=embed_model,
        table_name=EMBED_TABLE_NAME,
        connection_string=tidb_connection_string,
    )
    print("---Stored documents in Vector Store---")

# =============================================================================
# KNOWLEDGE GRAPH DATA MODELS
# =============================================================================

# Reset tables flag (set to True during development to recreate tables)
reset_tables = False
if reset_tables:
    db.execute("DROP TABLE IF EXISTS relationships;")
    db.execute("DROP TABLE IF EXISTS entities;")


class DBEntity(TableModel):
    """
    Database model for storing knowledge graph entities.
    
    Each entity represents a concept, person, event, or term extracted from documents.
    Entities are automatically embedded for semantic search.
    """
    __tablename__ = "entities"
    __table_args__ = {'extend_existing': True}

    id: int = Field(default=None, primary_key=True)
    name: str = Field(default=None)
    description: str = Field(default=None, sa_type=TEXT)
    embedding: List[float] = openai_embed.VectorField(source_field="entity_str")
    meta: dict = Field(sa_type=JSON, default_factory=dict)

    @property
    def entity_str(self) -> str:
        """Combined string representation used for embedding generation."""
        return f"{self.name} : {self.description}"


class DBRelationship(TableModel):
    """
    Database model for storing relationships between entities in the knowledge graph.
    
    Relationships describe how entities are connected and are also embedded
    for semantic search capabilities.
    """
    __tablename__ = "relationships"
    __table_args__ = {'extend_existing': True}
    
    id: int = Field(default=None, primary_key=True)
    source_entity_id: int = Field(foreign_key="entities.id")
    target_entity_id: int = Field(foreign_key="entities.id")
    relationship_desc: str = Field(default=None, sa_type=TEXT)
    embedding: Optional[List[float]] = openai_embed.VectorField(source_field="relationship_desc")
    
    # Foreign key relationships
    source_entity: DBEntity = SQLRelationship(
        sa_relationship_kwargs={
            "primaryjoin": f"DBRelationship.source_entity_id == DBEntity.id",
            "lazy": "joined",
        },
    )
    target_entity: DBEntity = SQLRelationship(
        sa_relationship_kwargs={
            "primaryjoin": f"DBRelationship.target_entity_id == DBEntity.id",
            "lazy": "joined",
        },
    )

    @property
    def source_name(self) -> str:
        """Name of the source entity in this relationship."""
        return self.source_entity.name
    
    @property
    def target_name(self) -> str:
        """Name of the target entity in this relationship."""
        return self.target_entity.name


# Create database tables
entity_table = db.create_table(schema=DBEntity, mode="exist_ok")
relationship_table = db.create_table(schema=DBRelationship, mode="exist_ok")

# =============================================================================
# KNOWLEDGE GRAPH EXTRACTION MODELS
# =============================================================================

# Configure DSPy with OpenAI
open_ai = dspy.LM(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=4096
)
dspy.settings.configure(lm=open_ai)


class Entity(BaseModel):
    """Pydantic model for entities extracted from text."""
    name: str = PyField(
        description="Name of the concept, person, event, or term. Should be concise and specific."
    )
    description: str = PyField(
        description=(
            "A clear and complete explanation of the entity. "
            "Example: 'Photosynthesis': 'Photosynthesis is the process by which green plants "
            "use sunlight to synthesize food from carbon dioxide and water.'"
        )
    )


class Relationship(BaseModel):
    """Pydantic model for relationships between entities."""
    source_name: str = PyField(
        description="Source entity name, must match one of the extracted entities."
    )
    target_name: str = PyField(
        description="Target entity name, must match one of the extracted entities."
    )
    relationship_desc: str = PyField(
        description=(
            "Complete sentence describing the relationship. "
            "Example: 'Newton's Laws of Motion form the foundation of Classical Mechanics.'"
        )
    )


class KnowledgeGraph(BaseModel):
    """Complete knowledge graph with entities and their relationships."""
    entities: List[Entity] = PyField(
        description="List of academic entities (concepts, people, events)."
    )
    relationships: List[Relationship] = PyField(
        description="List of relationships connecting entities."
    )


class ExtractGraphTriplet(dspy.Signature):
    """
    DSPy signature for extracting knowledge graphs from educational text.
    
    This signature defines the input/output format and provides detailed instructions
    for the LLM on how to extract meaningful entities and relationships.
    """
    text = dspy.InputField(
        desc="A passage from educational material (e.g., textbook, article, course notes)."
    )
    knowledge: KnowledgeGraph = dspy.OutputField(
        desc="Graph of concepts and relationships extracted from the text."
    )


class Extractor(dspy.Module):
    """DSPy module for knowledge graph extraction."""
    
    def __init__(self):
        super().__init__()
        self.prog_graph = dspy.Predict(ExtractGraphTriplet)
    
    def forward(self, text: str) -> KnowledgeGraph:
        """Extract knowledge graph from input text."""
        return self.prog_graph(text=text)


# Global extractor instance
kg_extractor = Extractor()


def extract_knowledge_graph(text: str) -> KnowledgeGraph:
    """
    Extract a knowledge graph from the given text using LLM.
    
    Args:
        text: Input text to extract knowledge from
    
    Returns:
        KnowledgeGraph object containing extracted entities and relationships
    """
    return kg_extractor(text=text).knowledge

# =============================================================================
# KNOWLEDGE GRAPH DATABASE OPERATIONS
# =============================================================================

def get_entities_by_names(names: List[str], metadata: Dict[str, Any] = None) -> List[DBEntity]:
    """
    Retrieve entities from database by their names and metadata filters.
    
    Args:
        names: List of entity names to search for
        metadata: Metadata filters (e.g., thread_id)
    
    Returns:
        List of matching DBEntity objects
    """
    if not names:
        return []
    
    if metadata is None:
        metadata = {}
    
    filters = {"name": {"$in": names}}
    if "thread_id" in metadata:
        filters["meta.thread_id"] = {"$eq": metadata["thread_id"]}
    
    return entity_table.query(filters=filters).to_pydantic()


def save_knowledge_graph(kg: KnowledgeGraph, metadata: Dict[str, Any] = None) -> None:
    """
    Save a knowledge graph to the database.
    
    This function:
    1. Checks for existing entities with the same names
    2. Creates new entities that don't exist
    3. Creates relationships between entities
    4. Handles database transactions safely
    
    Args:
        kg: KnowledgeGraph object to save
        metadata: Metadata to associate with entities (e.g., thread_id)
    """
    if metadata is None:
        metadata = {}
    
    with db.session() as session:
        # Find existing entities with matching names
        existing_entities = get_entities_by_names([e.name for e in kg.entities], metadata)
        entity_name_to_id = {entity.name: entity.id for entity in existing_entities}

        # Create new entities that don't exist yet
        entities_to_add = [
            DBEntity(name=e.name, description=e.description, meta=metadata)
            for e in kg.entities
            if e.name not in entity_name_to_id
        ]

        if entities_to_add:
            new_entities = entity_table.bulk_insert(entities_to_add)
            entity_name_to_id.update({e.name: e.id for e in new_entities})

        # Create relationships only if both source and target entities exist
        valid_relationships = [
            DBRelationship(
                source_entity_id=entity_name_to_id[r.source_name],
                target_entity_id=entity_name_to_id[r.target_name],
                relationship_desc=r.relationship_desc
            )
            for r in kg.relationships
            if r.source_name in entity_name_to_id and r.target_name in entity_name_to_id
        ]

        if valid_relationships:
            relationship_table.bulk_insert(valid_relationships)

        session.commit()
        print(f"---Saved {len(entities_to_add)} new entities and {len(valid_relationships)} relationships---")

# =============================================================================
# KNOWLEDGE GRAPH RETRIEVAL AND SEARCH
# =============================================================================

class RetrievedKnowledgeGraph(BaseModel):
    """Container for retrieved knowledge graph data."""
    entities: List[DBEntity]
    relationships: List[DBRelationship]


def retrieve_knowledge_graph(query: str, metadata: Dict[str, Any] = None, 
                            max_depth: int = 3, top_k: int = 10) -> Optional[RetrievedKnowledgeGraph]:
    """
    Retrieve relevant knowledge graph subgraph based on a query.
    
    Uses semantic search to find relevant starting entities, then performs
    breadth-first search to find connected entities and relationships.
    
    Args:
        query: Search query for finding relevant entities
        metadata: Metadata filters (e.g., thread_id)
        max_depth: Maximum depth for BFS traversal
        top_k: Maximum number of starting entities to find
    
    Returns:
        RetrievedKnowledgeGraph containing relevant entities and relationships,
        or None if no relevant entities found
    """
    if metadata is None:
        metadata = {}
    
    with db.session() as session:
        # Find starting entities using semantic search
        search_filters = {}
        if "thread_id" in metadata:
            search_filters["meta.thread_id"] = {"$eq": metadata["thread_id"]}
        
        start_entities = (entity_table.search(query)
                         .filter(search_filters)
                         .limit(top_k)
                         .to_pydantic())
        
        if not start_entities:
            return None

        # Perform BFS to find connected entities and relationships
        entities, relationships = knowledge_graph_bfs(session, start_entities, max_depth)
        
        return RetrievedKnowledgeGraph(
            entities=[DBEntity.model_validate(e) for e in entities],
            relationships=[DBRelationship.model_validate(r) for r in relationships]
        )


def get_connected_relationships(session, entity_id: int) -> List[DBRelationship]:
    """
    Get all relationships connected to a specific entity.
    
    Args:
        session: SQLAlchemy session
        entity_id: ID of the entity to find relationships for
    
    Returns:
        List of relationships where the entity is either source or target
    """
    stmt = (
        select(DBRelationship)
        .options(
            joinedload(DBRelationship.source_entity),
            joinedload(DBRelationship.target_entity),
        )
        .where(
            or_(
                DBRelationship.source_entity_id == entity_id,
                DBRelationship.target_entity_id == entity_id,
            )
        )
    )
    return session.execute(stmt).scalars().all()


def knowledge_graph_bfs(session, start_entities: List[DBEntity], max_depth: int) -> tuple[List[DBEntity], List[DBRelationship]]:
    """
    Perform breadth-first search from starting entities to find connected subgraph.
    
    Args:
        session: SQLAlchemy session
        start_entities: List of entities to start BFS from
        max_depth: Maximum depth to traverse
    
    Returns:
        Tuple of (entities, relationships) found during traversal
    """
    visited_entities = {e.id: e for e in start_entities}
    visited_relationships = {}

    # BFS queue: (entity, current_depth)
    queue = deque((e, 0) for e in start_entities)
    
    while queue:
        entity, depth = queue.popleft()
        
        if depth >= max_depth:
            continue

        # Find all relationships connected to this entity
        for rel in get_connected_relationships(session, entity.id):
            if rel.id in visited_relationships:
                continue
            
            visited_relationships[rel.id] = rel

            # Add unvisited neighboring entities to queue
            for neighbor in [rel.source_entity, rel.target_entity]:
                if neighbor.id not in visited_entities:
                    visited_entities[neighbor.id] = neighbor
                    queue.append((neighbor, depth + 1))

    return list(visited_entities.values()), list(visited_relationships.values())

# =============================================================================
# KNOWLEDGE GRAPH VISUALIZATION
# =============================================================================

class VisualizableEntity(Protocol):
    """Protocol for entities that can be visualized."""
    name: str
    description: str
    meta: Dict[str, Any]


class VisualizableRelationship(Protocol):
    """Protocol for relationships that can be visualized."""
    source_name: str
    target_name: str
    relationship_desc: str


class VisualizableKnowledgeGraph(Protocol):
    """Protocol for knowledge graphs that can be visualized."""
    entities: List[VisualizableEntity]
    relationships: List[VisualizableRelationship]


def visualize_knowledge_graph(kg: VisualizableKnowledgeGraph, filename: str,
                            custom_node_fn: Optional[Callable[[VisualizableEntity], dict]] = None) -> None:
    """
    Create an interactive HTML visualization of the knowledge graph.
    
    Args:
        kg: Knowledge graph to visualize
        filename: Output HTML filename
        custom_node_fn: Optional function to customize node appearance
    """
    net = Network(notebook=True, cdn_resources='remote')
    
    # Create mapping from entity names to node IDs
    node_name_to_id = {e.name: i for i, e in enumerate(kg.entities)}

    # Add nodes to the network
    for name, node_id in node_name_to_id.items():
        entity = kg.entities[node_id]
        node_properties = {}
        
        if custom_node_fn is not None:
            node_properties = custom_node_fn(entity)
        
        net.add_node(
            node_id, 
            label=entity.name, 
            title=entity.description, 
            **node_properties
        )

    # Add edges (relationships) to the network
    for relationship in kg.relationships:
        src_id = node_name_to_id.get(relationship.source_name)
        tgt_id = node_name_to_id.get(relationship.target_name)
        
        if src_id is not None and tgt_id is not None:
            net.add_edge(src_id, tgt_id, title=relationship.relationship_desc)

    # Generate filename if not provided
    if not filename or filename == "":
        filename = f"graph_{time.time()}.html"

    net.save_graph(filename)
    print(f"---Knowledge graph saved to {filename}---")

# =============================================================================
# ANSWER GENERATION WITH KNOWLEDGE GRAPH
# =============================================================================

class AnswerWithKG(dspy.Signature):
    """DSPy signature for answering questions using retrieved knowledge graph context."""
    knowledge_graph_context: str = dspy.InputField(
        desc="Formatted entities and relationships from knowledge graph"
    )
    question: str = dspy.InputField(desc="Question to answer")
    answer: str = dspy.OutputField(
        desc="Generated answer based on knowledge graph, formatted in Markdown. Say 'I don't know' if no context is provided"
    )


def answer_with_knowledge_graph(question: str, metadata: Dict[str, Any] = None, 
                              visualize_kg: bool = True) -> Dict[str, Any]:
    """
    Answer a question using knowledge graph context.
    
    Args:
        question: User's question
        metadata: Metadata filters for knowledge graph search
        visualize_kg: Whether to generate visualization
    
    Returns:
        Dictionary containing answer and knowledge graph path
    """
    if metadata is None:
        metadata = {}
    
    # Retrieve relevant knowledge graph
    kg = retrieve_knowledge_graph(question, metadata=metadata, max_depth=3)
    kg_path = None
    
    if kg:
        # Format knowledge graph context
        entities_str = "\n".join(f'{e.name}: {e.description}' for e in kg.entities)
        relationships_str = "\n".join(
            f'{r.source_name} -> {r.relationship_desc} -> {r.target_name}' 
            for r in kg.relationships
        )
        knowledge_graph_context = f"Entities:\n{entities_str}\n\nRelationships:\n{relationships_str}\n"

        # Generate visualization if requested
        if visualize_kg:
            kg_path = f"graph_{uuid.uuid4()}.html"
            visualize_knowledge_graph(kg, filename=".cache/graph/" + kg_path)
    else:
        knowledge_graph_context = ""

    # Generate answer using DSPy
    answer_question = dspy.Predict(AnswerWithKG)
    result = answer_question(
        knowledge_graph_context=knowledge_graph_context, 
        question=question
    )

    return {
        "answer": result.answer,
        "kg_path": kg_path
    }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clear_knowledge_graph() -> None:
    """
    Clear all knowledge graph data from the database.
    
    Warning: This will delete all entities and relationships!
    """
    with db.session() as session:
        session.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
        relationship_table.truncate()
        entity_table.truncate()
        session.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
        session.commit()
        print("---Knowledge graph cleared---")