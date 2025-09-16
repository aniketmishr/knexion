"""
Agentic RAG Workflow for Knexion

This module implements the core LangGraph workflow for the Knexion system.
It orchestrates a multi-step process that combines:
- Knowledge graph retrieval
- Vector-based document retrieval
- Document relevance grading
- Answer generation with hybrid context
- Answer quality validation
- Hallucination detection
- Web search integration for knowledge gaps

The workflow uses conditional routing to ensure high-quality, grounded responses
while falling back to web search when local knowledge is insufficient.
"""

import sqlite3
import uuid
from typing import List, TypedDict, Any, Dict, Annotated

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# Local imports
from llm_chains import generation_chain, retrieval_grader, answer_grader, hallucination_grader
from knowledge_store import create_vectorstore, retrieve_knowledge_graph, visualize_knowledge_graph

load_dotenv()

# =============================================================================
# WORKFLOW NODE IDENTIFIERS
# =============================================================================

START_NODE = "start_node"
END_NODE = "end_node"
RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
WEBSEARCH = "websearch"

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

class GraphState(TypedDict):
    """
    State object for the workflow containing all necessary data for processing.
    
    The state is passed between nodes and maintains context throughout
    the entire workflow execution.
    """
    question: str                              # User's original question
    generation: str                            # LLM-generated response
    web_search: bool                          # Flag indicating if web search is needed
    documents: List[Document]                 # Retrieved document context
    kg_str: str                              # Knowledge graph context as string
    kg_path: str                             # Path to knowledge graph visualization file
    messages: Annotated[List[BaseMessage], add_messages]  # Chat conversation history

# =============================================================================
# DATABASE SETUP AND CHECKPOINTING
# =============================================================================

# SQLite connection for conversation persistence
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# LangGraph checkpointer for conversation state management
checkpointer = SqliteSaver(conn=conn)

# Web search tool configuration
web_search_tool = TavilySearch(max_results=3)

# =============================================================================
# WORKFLOW NODE FUNCTIONS
# =============================================================================

def start_node(state: GraphState) -> GraphState:
    """
    Initialize the workflow with the user's question.
    
    Creates the initial message in the conversation history.
    
    Args:
        state: Current graph state
    
    Returns:
        Updated state with initial message
    """
    user_query = state["question"]
    state["messages"] = [HumanMessage(content=user_query)]
    return state


def retrieve(state: GraphState, config) -> Dict[str, Any]:
    """
    Retrieve context from both vector store and knowledge graph.
    
    This node performs hybrid retrieval:
    1. Retrieves relevant documents using vector similarity search
    2. Retrieves relevant knowledge graph subgraph using semantic search
    3. Formats knowledge graph context for downstream processing
    4. Generates knowledge graph visualization
    
    Args:
        state: Current graph state
        config: Configuration containing thread_id for filtering
    
    Returns:
        Dictionary with retrieved documents, KG context, and visualization path
    """
    print("---RETRIEVE---")
    
    # Extract configuration
    thread_id = config.get("configurable", {}).get("thread_id")
    question = state["question"]
    
    # Retrieve documents from vector store
    retriever = create_vectorstore(thread_id)
    documents = retriever.invoke(question)
    
    # Retrieve knowledge graph subgraph
    kg = retrieve_knowledge_graph(
        question, 
        metadata={"thread_id": thread_id}, 
        max_depth=3, 
        top_k=10
    )
    
    kg_path = None
    if kg:
        # Format knowledge graph context
        entities_str = "\n".join(f'{e.name}: {e.description}' for e in kg.entities)
        relationships_str = "\n".join(
            f'{r.source_name} -> {r.relationship_desc} -> {r.target_name}' 
            for r in kg.relationships
        )
        knowledge_graph_context = f"Entities:\n{entities_str}\n\nRelationships:\n{relationships_str}\n"
        
        # Generate visualization
        kg_path = f"graph_{uuid.uuid4()}.html"
        visualize_knowledge_graph(kg, filename=".cache/graph/" + kg_path)
    else:
        knowledge_graph_context = ""
    
    return {
        "documents": documents,
        "kg_str": knowledge_graph_context,
        "kg_path": kg_path,
        "question": question
    }


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Evaluate the relevance of retrieved documents to the user's question.
    
    Filters documents based on relevance scoring and sets a flag for web search
    if insufficient relevant documents are found.
    
    Args:
        state: Current graph state containing documents and question
    
    Returns:
        Dictionary with filtered documents and web search flag
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    web_search = False
    
    # Grade each document for relevance
    for doc in documents:
        score = retrieval_grader.invoke({
            "question": question,
            "document": doc.page_content
        })
        
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search
    }


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate an answer using knowledge graph and document context.
    
    Combines context from both the knowledge graph and retrieved documents
    to generate a comprehensive response.
    
    Args:
        state: Current graph state with question and context
    
    Returns:
        Dictionary with generated response
    """
    print("---GENERATE---")
    
    question = state["question"]
    documents = state["documents"]
    kg_str = state["kg_str"]
    
    # Generate answer using hybrid context
    generation = generation_chain.invoke({
        "kg_context": kg_str,
        "vector_context": documents,
        "question": question
    })
    
    return {
        "documents": documents,
        "question": question,
        "generation": generation
    }


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Perform web search to augment context with external information.
    
    Uses Tavily search to find relevant web content when local knowledge
    is insufficient to answer the question.
    
    Args:
        state: Current graph state
    
    Returns:
        Dictionary with updated document list including web search results
    """
    print("---WEB SEARCH---")
    
    question = state["question"]
    documents = state.get("documents", [])  # Get existing documents or empty list
    
    # Perform web search
    tavily_results = web_search_tool.invoke({"query": question})["results"]
    
    # Combine web results into a single document
    joined_tavily_result = "\n".join([
        tavily_result["content"] for tavily_result in tavily_results
    ])
    web_results = Document(page_content=joined_tavily_result)
    
    # Append web results to existing documents
    if documents:
        documents.append(web_results)
    else:
        documents = [web_results]
    
    return {
        "documents": documents,
        "question": question
    }


def end_node(state: GraphState) -> GraphState:
    """
    Finalize the workflow and prepare the response.
    
    Creates the final AI message with the generated response and associated
    metadata (knowledge graph path and document context).
    
    Args:
        state: Current graph state
    
    Returns:
        Updated state with final AI message
    """
    response = state["generation"]
    kg_path = state["kg_path"]
    docs_context = state["documents"]
    
    # Create final AI message with metadata
    state["messages"] = [AIMessage(
        content=response,
        additional_kwargs={
            "kg_path": kg_path,
            "docs": "\n\n".join([doc.page_content for doc in docs_context])
        }
    )]
    
    return state

# =============================================================================
# CONDITIONAL ROUTING FUNCTIONS
# =============================================================================

def decide_to_generate(state: GraphState) -> str:
    """
    Route to web search or answer generation based on document quality.
    
    Args:
        state: Current graph state
    
    Returns:
        Next node identifier (WEBSEARCH or GENERATE)
    """
    print("---ASSESS DOCUMENTS---")
    return WEBSEARCH if state["web_search"] else GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    """
    Evaluate the quality and groundedness of the generated answer.
    
    This function performs a two-stage evaluation:
    1. Hallucination check: Is the answer grounded in the provided context?
    2. Quality check: Does the answer adequately address the question?
    
    Args:
        state: Current graph state with generation and context
    
    Returns:
        Routing decision: "useful", "not useful", or "not supported"
    """
    print("---CHECK HALLUCINATIONS---")
    
    question = state["question"]
    documents = state["documents"]
    kg_str = state["kg_str"]
    generation = state["generation"]
    
    # Check if the answer is grounded in the provided context
    hallucination_score = hallucination_grader.invoke({
        "kg_context": kg_str,
        "vector_context": documents,
        "generation": generation
    })
    
    if hallucination_score.binary_score:
        # Answer is grounded, now check if it's useful
        print("---GENERATION GROUNDED IN DOCUMENTS---")
        
        quality_score = answer_grader.invoke({
            "question": question,
            "generation": generation
        })
        
        if quality_score.binary_score:
            print("---GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---GENERATION NOT GROUNDED IN DOCUMENTS---")
        return "not supported"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def retrieve_all_threads() -> List[str]:
    """
    Retrieve all conversation thread IDs from the checkpointer.
    
    Returns:
        List of thread IDs for existing conversations
    """
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        all_threads.add(thread_id)
    return list(all_threads)

# =============================================================================
# WORKFLOW CONSTRUCTION
# =============================================================================

# Create the state graph
workflow = StateGraph(GraphState)

# Add nodes to the workflow
workflow.add_node(START_NODE, start_node)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(END_NODE, end_node)

# Define the workflow entry point
workflow.set_entry_point(START_NODE)

# Add sequential edges
workflow.add_edge(START_NODE, RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(END_NODE, END)

# Add conditional edges for decision points
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,     # Regenerate if hallucinations detected
        "useful": END_NODE,            # End if answer is good
        "not useful": WEBSEARCH        # Search web for more context
    },
)

# Compile the workflow with checkpointing
agent = workflow.compile(checkpointer=checkpointer)

# =============================================================================
# WORKFLOW VISUALIZATION (Optional)
# =============================================================================

# Uncomment to generate workflow diagram
# agent.get_graph().draw_mermaid_png(output_file_path="workflow_graph.png")

# =============================================================================
# WORKFLOW EXECUTION HELPERS
# =============================================================================

def run_agent_query(question: str, thread_id: str) -> Dict[str, Any]:
    """
    Execute the complete agent workflow for a user question.
    
    Args:
        question: User's question
        thread_id: Conversation thread identifier
    
    Returns:
        Dictionary containing the final response and metadata
    """
    try:
        response = agent.invoke(
            {"question": question},
            config={'configurable': {'thread_id': thread_id}}
        )
        
        final_message = response['messages'][-1]
        return {
            "answer": final_message.content,
            "kg_path": final_message.additional_kwargs.get('kg_path'),
            "docs": final_message.additional_kwargs.get('docs')
        }
    except Exception as e:
        print(f"Error in agent execution: {e}")
        return {
            "answer": "I apologize, but I encountered an error processing your question.",
            "kg_path": None,
            "docs": None
        }


def get_conversation_history(thread_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve conversation history for a specific thread.
    
    Args:
        thread_id: Conversation thread identifier
    
    Returns:
        List of formatted message dictionaries
    """
    try:
        state = agent.get_state(
            config={'configurable': {'thread_id': thread_id}}
        )
        messages = state.values.get('messages', [])
        
        formatted_messages = []
        for message in messages:
            role = 'user' if isinstance(message, HumanMessage) else 'assistant'
            kg_path = message.additional_kwargs.get("kg_path")
            docs = message.additional_kwargs.get("docs")
            
            formatted_messages.append({
                'role': role,
                'content': message.content,
                'kg_path': kg_path,
                "docs": docs
            })
        
        return formatted_messages
    except Exception as e:
        print(f"Error retrieving conversation history: {e}")
        return []