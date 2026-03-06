import { VctrsDatabase } from './index.js';
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

const tmp = mkdtempSync(join(tmpdir(), 'vctrs-'));

try {
  // Test: create with dim/metric
  const db = new VctrsDatabase(join(tmp, 'testdb'), 4, 'cosine');

  db.add('a', [1, 0, 0, 0], { title: 'first' });
  db.add('b', [0, 1, 0, 0], { category: 'art' });
  db.add('c', [0, 0, 1, 0], { category: 'science' });
  console.log(`Added ${db.length} vectors`);

  // Test: search returns objects with .id, .distance, .metadata
  const results = db.search([0.9, 0.1, 0, 0]);
  console.log(`Search: ${results[0].id} (dist=${results[0].distance.toFixed(4)}), meta=${JSON.stringify(results[0].metadata)}`);

  // Test: upsert
  db.upsert('a', [0.5, 0.5, 0, 0], { title: 'updated' });
  const { metadata } = db.get('a');
  console.log(`Upsert: meta=${JSON.stringify(metadata)}`);

  db.upsert('d', [0, 0, 0, 1], { title: 'new' });
  console.log(`Upsert new: length=${db.length}`);

  // Test: filtered search
  const filtered = db.search([1, 0, 0, 0], 10, null, { category: 'science' });
  console.log(`Filtered search: ${filtered.map(r => r.id)} (only science)`);

  // Test: $ne filter
  const notArt = db.search([1, 0, 0, 0], 10, null, { category: { $ne: 'art' } });
  console.log(`$ne filter: ${notArt.map(r => r.id)} (not art)`);

  // Test: ids
  console.log(`IDs: ${db.ids().sort()}`);

  // Test: dim/metric getters
  console.log(`dim=${db.dim}, metric=${db.metric}`);

  // Test: save and auto-detect on reopen
  db.save();
  const db2 = new VctrsDatabase(join(tmp, 'testdb'));
  console.log(`Reopened: length=${db2.length}, dim=${db2.dim}, metric=${db2.metric}`);

  // Test: close (save alias)
  db2.close();

  console.log('\nAll tests passed!');
} finally {
  rmSync(tmp, { recursive: true });
}
