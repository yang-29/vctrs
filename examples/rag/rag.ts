/**
 * Minimal RAG (Retrieval-Augmented Generation) with vctrs + OpenAI.
 *
 * Index documents, retrieve relevant context, generate an answer.
 *
 * Usage:
 *   npm install vctrs @xenova/transformers openai
 *   export OPENAI_API_KEY=sk-...
 *   npx tsx rag.ts "What is HNSW?"
 */

import { VctrsDatabase } from "vctrs";
import { pipeline } from "@xenova/transformers";

// Chunks of text to index. In a real app these come from PDFs, web pages, etc.
const CHUNKS = [
  "HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest neighbor search. It builds a multi-layer graph where each layer is progressively sparser. Search starts at the top layer and greedily descends.",
  "Cosine similarity measures the angle between two vectors. It ranges from -1 (opposite) to 1 (identical). Cosine distance is 1 minus cosine similarity, so 0 means identical.",
  "Vector embeddings represent text, images, or other data as dense floating-point vectors. Similar items have similar embeddings. Models like sentence-transformers produce 384-dimensional embeddings.",
  "Brute-force search compares the query against every vector in the dataset. It gives perfect recall but is O(n). For large datasets, approximate methods like HNSW are much faster.",
  "Memory-mapped files let the OS manage which pages of a file are in RAM. This enables working with datasets larger than available memory. The OS pages data in and out as needed.",
  "RAG (Retrieval-Augmented Generation) combines a retriever (vector search) with a generator (LLM). The retriever finds relevant documents, which are passed as context to the LLM to generate accurate answers.",
];

async function main() {
  const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

  async function embed(texts: string | string[]): Promise<number[][]> {
    const output = await extractor(texts, { pooling: "mean", normalize: true });
    const data: number[] = Array.from(output.data as Float32Array);
    const dim = output.dims[output.dims.length - 1];
    const arr = Array.isArray(texts) ? texts : [texts];
    return arr.map((_, i) => data.slice(i * dim, (i + 1) * dim));
  }

  const sampleVec = await embed("test");
  const dim = sampleVec[0].length;

  const db = new VctrsDatabase("./rag_db", dim, "cosine");

  if (db.length === 0) {
    console.log("Indexing chunks...");
    const vectors = await embed(CHUNKS);
    const ids = CHUNKS.map((_, i) => `chunk${i}`);
    const metadata = CHUNKS.map((text) => ({ text }));
    db.addMany(ids, vectors, metadata);
    db.save();
  }

  const query = process.argv.slice(2).join(" ") || "What is HNSW and how does it work?";
  const queryVec = (await embed(query))[0];
  const results = db.search(queryVec, 3);
  const contextChunks = results.map((r) => r.metadata?.text as string);

  // Try OpenAI. If no API key, just print the retrieved context.
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.log("(No OPENAI_API_KEY set — showing retrieved context only)\n");
    console.log(`Query: ${query}\n`);
    console.log("Retrieved context:");
    for (const [i, chunk] of contextChunks.entries()) {
      console.log(`  ${i + 1}. ${chunk}`);
    }
    return;
  }

  const { default: OpenAI } = await import("openai");
  const client = new OpenAI();

  const context = contextChunks.join("\n\n");
  const response = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      { role: "system", content: `Answer based on this context:\n\n${context}` },
      { role: "user", content: query },
    ],
  });

  console.log(`Q: ${query}\n`);
  console.log(`A: ${response.choices[0].message.content}`);
}

main();
