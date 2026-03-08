/**
 * Semantic search over a collection of texts.
 *
 * Usage:
 *   npm install vctrs @xenova/transformers
 *   npx tsx search.ts
 */

import { VctrsDatabase } from "vctrs";
import { pipeline } from "@xenova/transformers";

const DOCS = [
  "Python is a high-level programming language known for its readability",
  "Rust provides memory safety without garbage collection",
  "JavaScript runs in the browser and on the server with Node.js",
  "PostgreSQL is a powerful open-source relational database",
  "Redis is an in-memory data store used for caching",
  "Docker containers package applications with their dependencies",
  "Kubernetes orchestrates container deployment at scale",
  "Git tracks changes in source code during development",
  "Linux is an open-source operating system kernel",
  "TCP/IP is the fundamental protocol suite of the internet",
  "Machine learning models learn patterns from training data",
  "Neural networks are inspired by biological brain structure",
  "GraphQL provides a query language for APIs",
  "WebAssembly enables near-native performance in web browsers",
  "SSH provides secure remote access to servers",
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

  const db = new VctrsDatabase("./search_db", dim, "cosine");

  if (db.length === 0) {
    console.log(`Indexing ${DOCS.length} documents...`);
    const vectors = await embed(DOCS);
    const ids = DOCS.map((_, i) => `doc${i}`);
    const metadata = DOCS.map((text) => ({ text }));
    db.addMany(ids, vectors, metadata);
    db.save();
    console.log("Done.\n");
  }

  const query = process.argv.slice(2).join(" ") || "database systems";
  console.log(`Search: "${query}"\n`);

  const queryVec = (await embed(query))[0];
  const results = db.search(queryVec, 5);

  for (const [rank, result] of results.entries()) {
    console.log(`  ${rank + 1}. [${result.distance.toFixed(3)}] ${result.metadata?.text}`);
  }
}

main();
