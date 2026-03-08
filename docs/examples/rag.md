# RAG (Retrieval-Augmented Generation)

Index documents, retrieve relevant context, generate an answer with an LLM.

=== "Python"

    ```python
    --8<-- "examples/rag/rag.py"
    ```

=== "TypeScript"

    ```typescript
    --8<-- "examples/rag/rag.ts"
    ```

## Run it

=== "Python"

    ```bash
    pip install vctrs sentence-transformers openai
    export OPENAI_API_KEY=sk-...
    python examples/rag/rag.py "What is HNSW?"
    ```

=== "TypeScript"

    ```bash
    npm install @yang-29/vctrs @xenova/transformers openai
    export OPENAI_API_KEY=sk-...
    npx tsx examples/rag/rag.ts "What is HNSW?"
    ```
