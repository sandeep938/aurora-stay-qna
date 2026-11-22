
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 4, vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks, model) -> str:

    rules = """
    1. GROUNDING & STRICT FAITHFULNESS: Answer the user's query using ONLY the provided Retrieved Policy Information. Do not extrapolate, guess, assume, or use any external or generic hotel knowledge. Every factual claim in your response must be directly supported by a retrieved chunk.
    2. COMPLETENESS & DETAIL RETENTION: Ensure all relevant specific details, conditions, exceptions, property codes (e.g., ASH-MIA, ASH-CHI, etc.), dollar amounts, fees, constraints, and limitations present in the retrieved chunks are fully included in the answer. Do not summarize away crucial specific information.
    3. NO INVENTIONS: Do not assume, invent, or formulate any operational rule, fee, restriction, or procedure that is not explicitly stated in the provided context.
    4. DIRECT RELEVANCY: Keep the response directly and specifically focused on answering the user's query. Address each specific question component clearly.
    5. INSUFFICIENT DETAILS: If the retrieved chunks are empty, or do not contain the specific policy, rule, or fee needed to answer the question, state exactly and only:
       "The available policy documents do not contain enough information to answer this question."
    6. RESOLVING CONFLICTS: If different retrieved chunks state conflicting rules, clearly point out the exact conflict as stated in the text.
    7. STRUCTURE & PROFESSIONALISM: Keep your response clean, professional, objective, and well-structured using markdown formatting.
    """

    prompt = f"""
    You are an expert, highly precise hotel policy assistant helping hotel representatives and guests.

    The retrieved information contains markdown tables. Match row values precisely to column headers. Do not mix up property codes (e.g., ASH-ATL with ASH-BOS) when reading across columns.

    If some parts of the user's query can be answered but other parts are missing from the retrieved context, answer the verifiable parts fully citing your source, and explicitly note which specific details were not found in the provided documents. Only fallback to 'The available policy documents do not contain enough information...' if no relevant information is present at all.

    Retrieved Policy Information:
    {retrieved_chunks}

    User Query:
    {query}

    Rules:
    {rules}

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 4,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k, vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
