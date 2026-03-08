# Interactive Demo

Try vctrs in the browser. This demo runs entirely client-side using WebAssembly — no server involved.

<iframe src="demo.html" width="100%" height="800" frameborder="0" style="border: 1px solid #e0e0e0; border-radius: 8px;"></iframe>

!!! tip "Try it"
    1. Click **Load Preset (colors)** to add 8 color vectors
    2. Click **Search** — the query `[0.9, 0.1, 0, 0]` finds red and orange as nearest neighbors
    3. Try adding your own vectors or searching with different queries

The WASM binary is ~220KB. Source: [`wasm/src/lib.rs`](https://github.com/yang-29/vctrs/blob/main/wasm/src/lib.rs).
