import { VctrsDatabase } from './index.js';
import { mkdtempSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import assert from 'assert';

let passed = 0;
let failed = 0;

function test(name, fn) {
  const tmp = mkdtempSync(join(tmpdir(), 'vctrs-'));
  try {
    fn(tmp);
    passed++;
    console.log(`  ✓ ${name}`);
  } catch (e) {
    failed++;
    console.log(`  ✗ ${name}: ${e.message}`);
  } finally {
    rmSync(tmp, { recursive: true });
  }
}

console.log('vctrs Node.js tests\n');

// -- Basic CRUD --

test('create and search', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 3, 'cosine');
  db.add('a', [1, 0, 0], { label: 'x' });
  db.add('b', [0, 1, 0]);
  db.add('c', [0, 0, 1]);

  assert.strictEqual(db.length, 3);
  assert.strictEqual(db.dim, 3);
  assert.strictEqual(db.metric, 'cosine');

  const results = db.search([0.9, 0.1, 0], 1);
  assert.strictEqual(results[0].id, 'a');
  assert.ok(results[0].distance >= 0);
  assert.deepStrictEqual(results[0].metadata, { label: 'x' });
});

test('get returns vector and metadata', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'euclidean');
  db.add('a', [1.0, 2.0], { k: 'v' });

  const result = db.get('a');
  assert.deepStrictEqual(result.vector, [1.0, 2.0]);
  assert.deepStrictEqual(result.metadata, { k: 'v' });
});

test('get nonexistent throws', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'cosine');
  assert.throws(() => db.get('missing'));
});

test('add duplicate throws', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'cosine');
  db.add('a', [1, 0]);
  assert.throws(() => db.add('a', [0, 1]));
});

test('add wrong dimension throws', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 3, 'cosine');
  assert.throws(() => db.add('a', [1, 0]));
});

test('search wrong dimension throws', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 3, 'cosine');
  db.add('a', [1, 0, 0]);
  assert.throws(() => db.search([1, 0]));
});

test('search empty database', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'cosine');
  const results = db.search([1, 0]);
  assert.strictEqual(results.length, 0);
});

// -- Upsert --

test('upsert insert and update', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'euclidean');
  db.upsert('a', [1, 0], { v: 1 });
  assert.strictEqual(db.length, 1);

  db.upsert('a', [0, 1], { v: 2 });
  assert.strictEqual(db.length, 1);

  const { vector, metadata } = db.get('a');
  assert.deepStrictEqual(vector, [0, 1]);
  assert.strictEqual(metadata.v, 2);
});

// -- Update --

test('update vector only', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'euclidean');
  db.add('a', [1, 0], { k: 'original' });
  db.update('a', [0, 1]);

  const { vector, metadata } = db.get('a');
  assert.deepStrictEqual(vector, [0, 1]);
  assert.strictEqual(metadata.k, 'original');
});

test('update metadata only', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'euclidean');
  db.add('a', [1, 0], { k: 'old' });
  db.update('a', null, { k: 'new' });

  const { vector, metadata } = db.get('a');
  assert.deepStrictEqual(vector, [1, 0]);
  assert.strictEqual(metadata.k, 'new');
});

test('update nonexistent throws', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'cosine');
  assert.throws(() => db.update('missing', [1, 0]));
});

// -- Delete --

test('delete removes from search', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'euclidean');
  db.add('a', [1, 0]);
  db.add('b', [0, 1]);
  db.add('c', [0.5, 0.5]);

  assert.strictEqual(db.delete('a'), true);
  assert.strictEqual(db.length, 2);
  assert.strictEqual(db.contains('a'), false);

  const results = db.search([1, 0], 10);
  assert.ok(results.every(r => r.id !== 'a'));
});

test('delete nonexistent returns false', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'cosine');
  assert.strictEqual(db.delete('missing'), false);
});

// -- Batch insert --

test('addMany', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'euclidean');
  db.addMany(
    ['a', 'b', 'c'],
    [[1, 0], [0, 1], [0.5, 0.5]],
    [{ k: 1 }, null, { k: 3 }]
  );
  assert.strictEqual(db.length, 3);

  const results = db.search([1, 0], 1);
  assert.strictEqual(results[0].id, 'a');
});

test('addMany length mismatch throws', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'cosine');
  assert.throws(() => db.addMany(['a'], [[1, 0], [0, 1]]));
});

// -- IDs and contains --

test('ids and contains', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'cosine');
  db.add('x', [1, 0]);
  db.add('y', [0, 1]);

  assert.strictEqual(db.contains('x'), true);
  assert.strictEqual(db.contains('z'), false);

  const ids = db.ids().sort();
  assert.deepStrictEqual(ids, ['x', 'y']);
});

// -- Filtered search --

test('filtered search eq', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'euclidean');
  db.add('a', [1, 0], { cat: 'sci' });
  db.add('b', [0.9, 0.1], { cat: 'art' });
  db.add('c', [0, 1], { cat: 'sci' });

  const results = db.search([1, 0], 10, null, { cat: 'sci' });
  assert.ok(results.every(r => r.metadata.cat === 'sci'));
  assert.strictEqual(results[0].id, 'a');
});

test('filtered search $ne', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'euclidean');
  db.add('a', [1, 0], { cat: 'x' });
  db.add('b', [0.9, 0.1], { cat: 'y' });

  const results = db.search([1, 0], 10, null, { cat: { $ne: 'x' } });
  assert.strictEqual(results.length, 1);
  assert.strictEqual(results[0].id, 'b');
});

test('filtered search $in', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'euclidean');
  db.add('a', [1, 0], { cat: 'x' });
  db.add('b', [0.9, 0.1], { cat: 'y' });
  db.add('c', [0, 1], { cat: 'z' });

  const results = db.search([1, 0], 10, null, { cat: { $in: ['x', 'z'] } });
  assert.strictEqual(results.length, 2);
  const ids = results.map(r => r.id).sort();
  assert.deepStrictEqual(ids, ['a', 'c']);
});

test('filtered search no matches', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'euclidean');
  db.add('a', [1, 0], { cat: 'x' });

  const results = db.search([1, 0], 10, null, { cat: 'z' });
  assert.strictEqual(results.length, 0);
});

// -- Search many --

test('searchMany basic', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'euclidean');
  db.add('a', [1, 0]);
  db.add('b', [0, 1]);

  const results = db.searchMany([[1, 0], [0, 1]], 1);
  assert.strictEqual(results.length, 2);
  assert.strictEqual(results[0][0].id, 'a');
  assert.strictEqual(results[1][0].id, 'b');
});

test('searchMany with filter', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'euclidean');
  db.add('a', [1, 0], { cat: 'x' });
  db.add('b', [0.9, 0.1], { cat: 'y' });
  db.add('c', [0, 1], { cat: 'x' });

  const results = db.searchMany([[1, 0], [0, 1]], 10, null, { cat: 'x' });
  assert.strictEqual(results.length, 2);
  for (const batch of results) {
    assert.ok(batch.every(r => r.metadata.cat === 'x'));
  }
});

// -- Persistence --

test('save and reopen', (tmp) => {
  const path = join(tmp, 'db');

  const db1 = new VctrsDatabase(path, 3, 'euclidean');
  db1.add('a', [1, 2, 3], { k: 'v' });
  db1.add('b', [4, 5, 6]);
  db1.save();

  // Reopen with auto-detect.
  const db2 = new VctrsDatabase(path);
  assert.strictEqual(db2.length, 2);
  assert.strictEqual(db2.dim, 3);
  assert.strictEqual(db2.metric, 'euclidean');

  const { metadata } = db2.get('a');
  assert.strictEqual(metadata.k, 'v');
});

test('close is save alias', (tmp) => {
  const path = join(tmp, 'db');
  const db = new VctrsDatabase(path, 2, 'cosine');
  db.add('a', [1, 0]);
  db.close();

  const db2 = new VctrsDatabase(path);
  assert.strictEqual(db2.length, 1);
});

// -- Compact --

test('compact reclaims slots', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'euclidean');
  db.add('a', [1, 0]);
  db.add('b', [0, 1]);
  db.add('c', [0.5, 0.5]);
  db.delete('b');

  assert.strictEqual(db.deletedCount, 1);
  assert.strictEqual(db.totalSlots, 3);

  db.compact();
  assert.strictEqual(db.length, 2);
  assert.strictEqual(db.deletedCount, 0);
  assert.strictEqual(db.totalSlots, 2);

  // Search still works.
  const results = db.search([1, 0], 1);
  assert.strictEqual(results[0].id, 'a');
});

test('compact then save and reload', (tmp) => {
  const path = join(tmp, 'db');
  const db = new VctrsDatabase(path, 2, 'euclidean');
  db.add('a', [1, 0], { k: 1 });
  db.add('b', [0, 1]);
  db.delete('b');
  db.compact();
  db.save();

  const db2 = new VctrsDatabase(path);
  assert.strictEqual(db2.length, 1);
  assert.strictEqual(db2.totalSlots, 1);
  assert.strictEqual(db2.contains('a'), true);
  assert.strictEqual(db2.contains('b'), false);
});

// -- Quantized search --

test('quantized search', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 3, 'cosine');
  db.add('a', [1, 0, 0]);
  db.add('b', [0, 1, 0]);
  db.add('c', [0, 0, 1]);

  assert.strictEqual(db.quantizedSearch, false);
  db.enableQuantizedSearch();
  assert.strictEqual(db.quantizedSearch, true);

  const results = db.search([1, 0, 0], 1);
  assert.strictEqual(results[0].id, 'a');

  db.disableQuantizedSearch();
  assert.strictEqual(db.quantizedSearch, false);
});

test('quantized search loads from disk', (tmp) => {
  const path = join(tmp, 'db');
  const db = new VctrsDatabase(path, 2, 'cosine', null, null, true);
  db.add('a', [1, 0]);
  db.add('b', [0, 1]);
  db.save();

  const db2 = new VctrsDatabase(path);
  assert.strictEqual(db2.quantizedSearch, true);
  const results = db2.search([1, 0], 1);
  assert.strictEqual(results[0].id, 'a');
});

// -- Metrics --

test('euclidean metric', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'euclidean');
  db.add('a', [0, 0]);
  db.add('b', [3, 4]);

  const results = db.search([0, 0], 1);
  assert.strictEqual(results[0].id, 'a');
  assert.ok(results[0].distance < 0.001);
});

test('dot_product metric', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'dot_product');
  db.add('a', [1, 0]);
  db.add('b', [0, 1]);

  const results = db.search([1, 0], 1);
  assert.strictEqual(results[0].id, 'a');
});

test('l2 metric alias', (tmp) => {
  const db = new VctrsDatabase(join(tmp, 'db'), 2, 'l2');
  assert.strictEqual(db.metric, 'euclidean');
});

// -- Summary --

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
