# Interactive Demo

Try vctrs in the browser. Everything below runs client-side via WebAssembly — no server, no network requests.

<iframe src="../demo.html" width="100%" height="1400" frameborder="0" style="border: 1px solid #e0e0e0; border-radius: 8px;"></iframe>

## Two modes

### Manual Vectors (default)
Pick a preset (Movies, Cities, or Foods) — each loads 20 items as 8-dimensional vectors where each dimension represents a quality. Click suggested searches to see how cosine similarity finds related items.

### Semantic Search (AI)
Click **Semantic Search (AI)** to enable real natural-language search powered by [mdbr-leaf-ir](https://huggingface.co/MongoDB/mdbr-leaf-ir) — the #1 ranked retrieval model under 100M params on MTEB — running entirely in your browser. The model (~30MB) is downloaded once and cached.

Load a preset (Sentences, Products, or Questions), then search with plain English — the model embeds your query into 768 dimensions and vctrs finds the closest matches.

!!! note "Everything runs locally"
    Both the vctrs WASM binary (~220KB) and the transformer model run entirely in your browser. No data is sent to any server.

Source: [`wasm/src/lib.rs`](https://github.com/yang-29/vctrs/blob/main/wasm/src/lib.rs)
