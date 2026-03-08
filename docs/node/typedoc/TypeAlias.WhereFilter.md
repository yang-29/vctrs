[vctrs](README.md) / WhereFilter

```ts
type WhereFilter = Record<string, any>;
```

Defined in: [index.d.ts:37](https://github.com/yang-29/vctrs/blob/1ac8fc97d294178870f629cdcbed183d98989fb4/node/index.d.ts#L37)

Metadata filter for search queries.

Supports equality, operators, and compound filters:
- Equality: `{ field: "value" }`
- Not-equal: `{ field: { $ne: "value" } }`
- In-set: `{ field: { $in: ["a", "b"] } }`
- Numeric ranges: `{ field: { $gt: 10, $lte: 20 } }`
- Compound (AND): `{ f1: "v1", f2: { $gt: 5 } }`
