[vctrs](README.md) / VctrsStats

Defined in: [index.d.ts:40](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L40)

Graph-level statistics returned by `stats()`.

## Properties

| Property | Type | Description | Defined in |
| ------ | ------ | ------ | ------ |
| <a id="avgdegreelayer0"></a> `avgDegreeLayer0` | `number` | Average degree (edges) on layer 0. | [index.d.ts:48](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L48) |
| <a id="maxdegreelayer0"></a> `maxDegreeLayer0` | `number` | Maximum degree on layer 0. | [index.d.ts:50](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L50) |
| <a id="memorygraphbytes"></a> `memoryGraphBytes` | `number` | Memory used by graph structure in bytes. | [index.d.ts:56](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L56) |
| <a id="memoryquantizedbytes"></a> `memoryQuantizedBytes` | `number` | Memory used by quantized vectors in bytes (0 if not quantized). | [index.d.ts:58](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L58) |
| <a id="memoryvectorsbytes"></a> `memoryVectorsBytes` | `number` | Memory used by vector storage in bytes. | [index.d.ts:54](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L54) |
| <a id="mindegreelayer0"></a> `minDegreeLayer0` | `number` | Minimum degree on layer 0. | [index.d.ts:52](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L52) |
| <a id="numdeleted"></a> `numDeleted` | `number` | Number of deleted vectors not yet reclaimed. | [index.d.ts:44](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L44) |
| <a id="numlayers"></a> `numLayers` | `number` | Number of HNSW layers. | [index.d.ts:46](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L46) |
| <a id="numvectors"></a> `numVectors` | `number` | Number of active (non-deleted) vectors. | [index.d.ts:42](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L42) |
| <a id="usesbruteforce"></a> `usesBruteForce` | `boolean` | Whether brute-force search is being used (small datasets). | [index.d.ts:60](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L60) |
| <a id="usesquantizedsearch"></a> `usesQuantizedSearch` | `boolean` | Whether quantized HNSW traversal is enabled. | [index.d.ts:62](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L62) |
