# Interactive Demo

Try vctrs in the browser. Everything below runs client-side via WebAssembly — no server, no network requests.

<iframe src="../demo.html" width="100%" height="900" frameborder="0" style="border: 1px solid #e0e0e0; border-radius: 8px;"></iframe>

!!! tip "How to use"
    1. **Pick a preset** — Movies, Cities, or Foods. Each loads 20 items as 8-dimensional vectors.
    2. **Click a suggested search** — each one explains what it searches for and why those results appear.
    3. **Experiment** — edit the query vector, change k, add your own vectors, or delete items and re-search.

    The 8 dimensions represent different qualities (e.g. for Movies: action, comedy, drama, sci-fi, romance, horror, animation, indie). Higher values = more of that quality. Cosine similarity finds vectors pointing in the same direction regardless of magnitude.

The WASM binary is ~220KB. Source: [`wasm/src/lib.rs`](https://github.com/yang-29/vctrs/blob/main/wasm/src/lib.rs).
