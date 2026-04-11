"""Orchestration logic for vision + RAG.

Integrates the vision model predictions with RAG-based knowledge retrieval
and Groq LLM for generating diagnostic reports and treatment recommendations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.utils import get_groq_api_key

# Default model configurations
DEFAULT_MODEL = "llama-3.1-70b-versatile"
FAST_MODEL = "llama-3.1-8b-instant"


def get_groq_llm(model_name: str = DEFAULT_MODEL, temperature: float = 0.3) -> ChatGroq:
    """Initialize Groq LLM client.

    Args:
        model_name: Groq model to use (default: llama-3.1-70b-versatile).
            Use llama-3.1-8b-instant for faster inference.
        temperature: Sampling temperature (0.0-1.0). Lower for more deterministic output.

    Returns:
        Configured ChatGroq instance.

    Raises:
        ValueError: If LLM_API is not configured.
    """
    api_key = get_groq_api_key()
    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=2048,
    )


def create_diagnostic_prompt() -> ChatPromptTemplate:
    """Create the diagnostic prompt template for agronomist AI.

    The prompt instructs the LLM to act as an expert agronomist and generate
treatment recommendations based on disease classification and retrieved context.

    Returns:
        ChatPromptTemplate with system and human messages.
    """
    system_template = (
        "You are an expert agronomist specializing in plant disease diagnosis and treatment. "
        "Your task is to analyze the detected plant disease and provide clear, actionable "
        "treatment recommendations based on agricultural best practices and the provided "
        "reference materials. Be concise but thorough. Focus on practical, evidence-based "
        "solutions that farmers can implement."
    )

    human_template = (
        "DISEASE DETECTED: {class_name}\n\n"
        "CONFIDENCE SCORE: {confidence:.1%}\n\n"
        "RETRIEVED CONTEXT:\n{context}\n\n"
        "Based on the disease classification and the reference materials above, provide:\n"
        "1. Brief disease description (2-3 sentences)\n"
        "2. Immediate treatment steps (priority actions)\n"
        "3. Preventive measures for future crops\n"
        "4. When to seek professional extension services\n\n"
        "Format your response in clear, actionable bullet points suitable for farmers."
    )

    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", human_template),
    ])


def format_retrieved_docs(docs: Sequence) -> str:
    """Format retrieved documents into context string.

    Args:
        docs: List of retrieved LangChain Document objects.

    Returns:
        Formatted context string with document sources.
    """
    if not docs:
        return "No reference materials found for this condition."

    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content[:500]  # Limit content length
        formatted.append(
            f"[Reference {i}] Source: {source} (Page: {page})\n{content}\n"
        )

    return "\n---\n".join(formatted)


def generate_diagnostic_report(
    class_name: str,
    confidence: float,
    retrieved_docs: Sequence,
    model_name: str = DEFAULT_MODEL,
) -> str:
    """Generate diagnostic report using Groq LLM.

    Combines vision model prediction with RAG-retrieved context to generate
    comprehensive treatment recommendations via Groq LLM.

    Args:
        class_name: Predicted disease class name from vision model.
        confidence: Model confidence score (0.0-1.0).
        retrieved_docs: Retrieved documents from ChromaDB vector store.
        model_name: Groq model to use (default: llama3-70b-8192).

    Returns:
        Generated diagnostic report as markdown-formatted string.

    Raises:
        ValueError: If GROQ_API_KEY is not configured.
        RuntimeError: If LLM invocation fails.
    """
    # Initialize components
    llm = get_groq_llm(model_name=model_name)
    prompt = create_diagnostic_prompt()
    context = format_retrieved_docs(retrieved_docs)

    # Build and execute chain
    chain = prompt | llm | StrOutputParser()

    try:
        response = chain.invoke({
            "class_name": class_name,
            "confidence": confidence,
            "context": context,
        })
    except Exception as e:
        raise RuntimeError(f"Failed to generate diagnostic report: {e}") from e

    return response


def run_dry_run_test() -> None:
    """Test Groq integration with simulated prediction.

    Performs a dry run to verify:
    - GROQ_API_KEY is properly loaded
    - Groq API is accessible and responsive
    - LLM can generate treatment recommendations

    Prints timing metrics and sample output.
    """
    import time

    print("🧪 Running Groq Integration Dry Run Test")
    print("=" * 50)

    # Simulate a vision model prediction
    test_class = "Tomato_Late_blight"
    test_confidence = 0.94

    # Create mock retrieved context (in production, this comes from ChromaDB)
    class MockDoc:
        def __init__(self, content: str, source: str, page: int) -> None:
            self.page_content = content
            self.metadata = {"source": source, "page": page}

    mock_docs = [
        MockDoc(
            "Late blight is a devastating disease caused by Phytophthora infestans. "
            "Symptoms include water-soaked lesions on leaves that turn brown/black. "
            "Fungicides containing chlorothalonil or mancozeb are effective treatments.",
            "Tomato_Disease.pdf",
            12,
        ),
        MockDoc(
            "For late blight management, remove and destroy infected plant debris. "
            "Apply preventive fungicides before symptoms appear during wet weather.",
            "MODULE3.pdf",
            45,
        ),
    ]

    # Test with fast model for speed check
    print(f"Test case: {test_class} (confidence: {test_confidence:.1%})")
    print(f"Model: {FAST_MODEL} (for quick validation)")
    print("-" * 50)

    start_time = time.time()

    try:
        report = generate_diagnostic_report(
            class_name=test_class,
            confidence=test_confidence,
            retrieved_docs=mock_docs,
            model_name=FAST_MODEL,
        )
        elapsed = time.time() - start_time

        print(f"✅ Success! Response time: {elapsed:.2f}s")
        print("\n📋 Sample Output:")
        print(report[:500] + "..." if len(report) > 500 else report)

    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        raise
    except Exception as e:
        print(f"❌ Runtime Error: {e}")
        raise


def main() -> None:
    """CLI entry point for orchestrator testing."""
    run_dry_run_test()


if __name__ == "__main__":
    main()
