"""
Agent Utilities for Knexion

This module provides utility functions and LLM chains for the Knexion agent system.
It includes:
- LLM model configurations (Google Gemini and OpenAI embeddings)
- Grading chains for answer quality assessment
- Document relevance evaluation
- Hallucination detection
- Answer generation with hybrid context (Knowledge Graph + Vector Search)

The module uses structured outputs and chain compositions to ensure reliable
and validated responses from the AI system.
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

load_dotenv()

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# Primary LLM for response generation and evaluation tasks
llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0  # Deterministic responses for consistency
)

# Embedding model for vector similarity search
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")

# =============================================================================
# STRUCTURED OUTPUT MODELS
# =============================================================================

class GradeAnswer(BaseModel):
    """
    Structured output model for answer quality grading.
    
    Evaluates whether an AI-generated answer adequately addresses
    the user's question.
    """
    binary_score: bool = Field(
        description="Whether the answer addresses the question: True for 'yes', False for 'no'"
    )


class GradeHallucinations(BaseModel):
    """
    Structured output model for hallucination detection.
    
    Evaluates whether an AI response is grounded in the provided
    context or contains unsupported claims.
    """
    binary_score: bool = Field(
        description="Whether the answer is grounded in facts: True for 'yes', False for 'no'"
    )


class GradeDocuments(BaseModel):
    """
    Structured output model for document relevance assessment.
    
    Evaluates whether retrieved documents are relevant to the user's question.
    """
    binary_score: str = Field(
        description="Whether documents are relevant to the question: 'yes' or 'no'"
    )

# =============================================================================
# ANSWER QUALITY GRADING CHAIN
# =============================================================================

# Create structured LLM for answer grading
structured_llm_answer_grader = llm_model.with_structured_output(GradeAnswer)

# System prompt for answer quality assessment
grade_answer_system = """You are a grader assessing whether an answer addresses and resolves a user's question.

Evaluation criteria:
- Does the answer directly respond to what was asked?
- Does it provide sufficient information to resolve the question?
- Is the response relevant and on-topic?

Give a binary score: 'yes' means the answer adequately resolves the question, 'no' means it does not."""

# Create prompt template for answer grading
grade_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", grade_answer_system),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
])

# Complete answer grading chain
answer_grader: RunnableSequence = grade_answer_prompt | structured_llm_answer_grader

# =============================================================================
# HYBRID RAG ANSWER GENERATION CHAIN
# =============================================================================

# Prompt template for generating answers using both KG and vector context
hybrid_rag_prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks using a hybrid retrieval system.

Use the following pieces of retrieved context to answer the question:
- Knowledge Graph Context: Structured entities and relationships
- Vector Search Context: Relevant document passages

Instructions:
- Synthesize information from both sources when available
- If you don't know the answer based on the provided context, clearly state that
- Keep answers concise but comprehensive (maximum three sentences)
- Prioritize accuracy over completeness

Question: {question}

Knowledge Graph Context:
{kg_context}

Vector Search Context:
{vector_context}

Answer:"""
)

# Complete generation chain
generation_chain = hybrid_rag_prompt | llm_model | StrOutputParser()

# =============================================================================
# HALLUCINATION DETECTION CHAIN
# =============================================================================

# Create structured LLM for hallucination grading
structured_llm_hallucination_grader = llm_model.with_structured_output(GradeHallucinations)

# System prompt for hallucination detection
grade_hallucination_system = """You are a grader assessing whether an LLM generation is grounded in and supported by the provided facts from a hybrid RAG system.

The hybrid system uses:
- Knowledge Graph: Structured entities and relationships
- Vector Database: Relevant document passages

Evaluation criteria:
- Is every claim in the answer supported by the provided context?
- Are there any contradictions with the given facts?
- Does the answer make unsupported inferences or assumptions?

Give a binary score:
- 'yes': The answer is fully grounded in and supported by the provided facts
- 'no': The answer contains hallucinations, contradictions, or unsupported claims"""

# Create prompt template for hallucination detection
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", grade_hallucination_system),
    ("human", """Knowledge Graph Facts:
{kg_context}

Vector Search Facts:
{vector_context}

LLM Generation: {generation}"""),
])

# Complete hallucination grading chain
hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_hallucination_grader

# =============================================================================
# DOCUMENT RELEVANCE GRADING CHAIN
# =============================================================================

# Create structured LLM for document grading
structured_llm_doc_grader = llm_model.with_structured_output(GradeDocuments)

# System prompt for document relevance assessment
doc_grader_system = """You are a grader assessing the relevance of retrieved documents to a user question.

Evaluation criteria:
- Does the document contain keywords related to the question?
- Does the document have semantic meaning related to the question topic?
- Could the information in this document help answer the question?

A document is considered relevant if it contains any information that could potentially help answer the question, even if it doesn't contain the complete answer.

Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

# Create prompt template for document grading
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", doc_grader_system),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])

# Complete document relevance grading chain
retrieval_grader = grade_prompt | structured_llm_doc_grader

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def evaluate_answer_quality(question: str, generation: str) -> bool:
    """
    Evaluate whether a generated answer adequately addresses the question.
    
    Args:
        question: The original user question
        generation: The AI-generated answer
    
    Returns:
        True if the answer adequately addresses the question, False otherwise
    """
    result = answer_grader.invoke({
        "question": question,
        "generation": generation
    })
    return result.binary_score


def check_hallucination(kg_context: str, vector_context: str, generation: str) -> bool:
    """
    Check if a generated answer contains hallucinations or unsupported claims.
    
    Args:
        kg_context: Knowledge graph context string
        vector_context: Vector search context string  
        generation: The AI-generated answer
    
    Returns:
        True if the answer is grounded in the context, False if it contains hallucinations
    """
    result = hallucination_grader.invoke({
        "kg_context": kg_context,
        "vector_context": vector_context,
        "generation": generation
    })
    return result.binary_score


def assess_document_relevance(document: str, question: str) -> bool:
    """
    Assess whether a retrieved document is relevant to the user's question.
    
    Args:
        document: The document content to evaluate
        question: The user's question
    
    Returns:
        True if the document is relevant, False otherwise
    """
    result = retrieval_grader.invoke({
        "document": document,
        "question": question
    })
    return result.binary_score.lower() == "yes"


def generate_hybrid_answer(question: str, kg_context: str, vector_context: str) -> str:
    """
    Generate an answer using both knowledge graph and vector search context.
    
    Args:
        question: The user's question
        kg_context: Knowledge graph context string
        vector_context: Vector search context string
    
    Returns:
        Generated answer string
    """
    return generation_chain.invoke({
        "question": question,
        "kg_context": kg_context,
        "vector_context": vector_context
    })


# =============================================================================
# CHAIN VALIDATION HELPERS
# =============================================================================

def validate_generation_pipeline(question: str, kg_context: str, vector_context: str) -> dict:
    """
    Complete validation pipeline for answer generation.
    
    Generates an answer and validates it for quality and hallucinations.
    
    Args:
        question: User's question
        kg_context: Knowledge graph context
        vector_context: Vector search context
    
    Returns:
        Dictionary containing:
        - answer: Generated answer
        - is_quality: Whether answer addresses the question
        - is_grounded: Whether answer is free from hallucinations
        - needs_web_search: Whether additional context is needed
    """
    # Generate answer
    answer = generate_hybrid_answer(question, kg_context, vector_context)
    
    # Validate answer quality
    is_quality = evaluate_answer_quality(question, answer)
    
    # Check for hallucinations
    is_grounded = check_hallucination(kg_context, vector_context, answer)
    
    # Determine if web search is needed
    needs_web_search = not (is_quality and is_grounded)
    
    return {
        "answer": answer,
        "is_quality": is_quality,
        "is_grounded": is_grounded,
        "needs_web_search": needs_web_search
    }


def batch_grade_documents(documents: list, question: str) -> dict:
    """
    Grade multiple documents for relevance to a question.
    
    Args:
        documents: List of document contents to grade
        question: User's question
    
    Returns:
        Dictionary containing:
        - relevant_docs: List of relevant documents
        - irrelevant_docs: List of irrelevant documents
        - relevance_scores: List of boolean scores for each document
    """
    relevant_docs = []
    irrelevant_docs = []
    relevance_scores = []
    
    for doc in documents:
        is_relevant = assess_document_relevance(doc, question)
        relevance_scores.append(is_relevant)
        
        if is_relevant:
            relevant_docs.append(doc)
        else:
            irrelevant_docs.append(doc)
    
    return {
        "relevant_docs": relevant_docs,
        "irrelevant_docs": irrelevant_docs,
        "relevance_scores": relevance_scores
    }