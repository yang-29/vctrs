/**
 * Find near-duplicate items using vector similarity.
 *
 * Usage:
 *   npm install vctrs @xenova/transformers
 *   npx tsx dedup.ts
 */

import { VctrsDatabase } from "vctrs";
import { pipeline } from "@xenova/transformers";

const ITEMS = [
  "How to reset my password",
  "I forgot my password, how do I reset it",
  "Password reset instructions",
  "How to change my email address",
  "Update my email address",
  "I want to change the email on my account",
  "What are your business hours",
  "When are you open",
  "What time do you close",
  "How do I cancel my subscription",
  "Cancel my account",
  "I want to stop my subscription",
  "Do you offer refunds",
  "Can I get my money back",
  "Refund policy",
  "How to contact support",
  "I need help from a human",
  "The app crashes when I open settings",
  "Settings page is broken, app crashes",
];

async function main() {
  const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

  async function embed(texts: string[]): Promise<number[][]> {
    const output = await extractor(texts, { pooling: "mean", normalize: true });
    const data: number[] = Array.from(output.data as Float32Array);
    const dim = output.dims[output.dims.length - 1];
    return texts.map((_, i) => data.slice(i * dim, (i + 1) * dim));
  }

  const vectors = await embed(ITEMS);
  const dim = vectors[0].length;

  const db = new VctrsDatabase("./dedup_db", dim, "cosine");
  const ids = ITEMS.map((_, i) => `item${i}`);
  const metadata = ITEMS.map((text) => ({ text }));
  db.addMany(ids, vectors, metadata);

  // Find clusters of near-duplicates.
  const threshold = 0.3; // cosine distance — lower means more similar
  const seen = new Set<number>();
  const clusters: Array<[number, Array<[number, number]>]> = [];

  for (let i = 0; i < ITEMS.length; i++) {
    if (seen.has(i)) continue;

    const results = db.search(vectors[i], 5);
    const cluster: Array<[number, number]> = [];

    for (const result of results) {
      const j = parseInt(result.id.replace("item", ""));
      if (j !== i && result.distance < threshold && !seen.has(j)) {
        cluster.push([j, result.distance]);
        seen.add(j);
      }
    }

    if (cluster.length > 0) {
      seen.add(i);
      clusters.push([i, cluster]);
    }
  }

  console.log(`Found ${clusters.length} duplicate clusters:\n`);
  for (const [i, dupes] of clusters) {
    console.log(`  "${ITEMS[i]}"`);
    for (const [j, dist] of dupes) {
      console.log(`    ≈ "${ITEMS[j]}" (distance: ${dist.toFixed(3)})`);
    }
    console.log();
  }
}

main();
