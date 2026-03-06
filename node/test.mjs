import { VctrsDatabase } from './index.js';
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

const tmp = mkdtempSync(join(tmpdir(), 'vctrs-'));

try {
  const db = new VctrsDatabase(join(tmp, 'testdb'), 4, 'cosine');

  // Add vectors
  db.add('a', [1, 0, 0, 0], { title: 'first' });
  db.add('b', [0, 1, 0, 0], { title: 'second' });
  db.add('c', [0, 0, 1, 0]);
  console.log(`Added ${db.length} vectors`);

  // Batch insert
  const ids = Array.from({ length: 100 }, (_, i) => `v${i}`);
  const vectors = ids.map(() => Array.from({ length: 4 }, () => Math.random()));
  db.addMany(ids, vectors);
  console.log(`After batch: ${db.length} vectors`);

  // Search
  const results = db.search([0.9, 0.1, 0, 0], 3);
  console.log('Search results:', results.map(r => `${r.id} (${r.distance.toFixed(4)})`));

  // Get
  const { vector, metadata } = db.get('a');
  console.log(`Get 'a': vector=[${vector.map(v => v.toFixed(1))}], meta=${JSON.stringify(metadata)}`);

  // Contains
  console.log(`Contains 'a': ${db.contains('a')}, contains 'z': ${db.contains('z')}`);

  // Delete
  db.delete('b');
  console.log(`After delete: ${db.length} vectors, contains 'b': ${db.contains('b')}`);

  // Update
  db.update('a', [0, 0, 0, 1], { title: 'updated' });
  const updated = db.get('a');
  console.log(`Updated 'a': meta=${JSON.stringify(updated.metadata)}`);

  // Save and reload
  db.save();
  const db2 = new VctrsDatabase(join(tmp, 'testdb'), 4, 'cosine');
  console.log(`Reloaded: ${db2.length} vectors`);

  const results2 = db2.search([1, 0, 0, 0], 1);
  console.log(`Search after reload: ${results2[0].id}`);

  console.log('\nAll tests passed!');
} finally {
  rmSync(tmp, { recursive: true });
}
