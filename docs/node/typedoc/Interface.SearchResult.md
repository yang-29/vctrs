[vctrs](README.md) / SearchResult

Defined in: [index.d.ts:7](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L7)

A single search result.

## Properties

| Property | Type | Description | Defined in |
| ------ | ------ | ------ | ------ |
| <a id="distance"></a> `distance` | `number` | Distance between the query and this vector. Lower is more similar for cosine and euclidean metrics. | [index.d.ts:14](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L14) |
| <a id="id"></a> `id` | `string` | The unique string identifier of the matched vector. | [index.d.ts:9](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L9) |
| <a id="metadata"></a> `metadata?` | `Record`\<`string`, `any`\> | The metadata object attached to this vector, or undefined. | [index.d.ts:16](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L16) |
