/// vctrs-server: thin HTTP wrapper around vctrs.
///
/// Usage:
///   vctrs-server --data ./data --port 8080
///
/// Endpoints:
///   POST   /collections                          — create collection
///   GET    /collections                          — list collections
///   DELETE /collections/:name                    — delete collection
///   POST   /collections/:name/add               — add vector
///   POST   /collections/:name/upsert            — upsert vector
///   POST   /collections/:name/add_many          — batch add
///   POST   /collections/:name/search            — search
///   GET    /collections/:name/get/:id            — get vector
///   DELETE /collections/:name/delete/:id         — delete vector
///   POST   /collections/:name/save              — persist to disk
///   GET    /collections/:name/stats              — graph stats

use axum::{
    extract::{Json, Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Router,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use tower_http::cors::CorsLayer;
use vctrs_core::client::Client;
use vctrs_core::db::{Database, HnswConfig};
use vctrs_core::distance::Metric;
use vctrs_core::filter::Filter;

#[derive(Parser)]
#[command(name = "vctrs-server", about = "HTTP server for vctrs vector database")]
struct Cli {
    /// Root data directory for collections.
    #[arg(long, default_value = "./vctrs_data")]
    data: String,

    /// Port to listen on.
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Host to bind to.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
}

struct AppState {
    client: Client,
    collections: RwLock<HashMap<String, Arc<Database>>>,
}

impl AppState {
    fn get_or_open(&self, name: &str) -> Result<Arc<Database>, AppError> {
        // Fast path: already cached.
        if let Some(db) = self.collections.read().get(name) {
            return Ok(Arc::clone(db));
        }

        // Slow path: open and cache.
        let db = self.client.get_collection(name)
            .map_err(|e| AppError(StatusCode::NOT_FOUND, e.to_string()))?;
        let db = Arc::new(db);
        self.collections.write().insert(name.to_string(), Arc::clone(&db));
        Ok(db)
    }
}

type AppResult<T> = Result<T, AppError>;

struct AppError(StatusCode, String);

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let body = serde_json::json!({ "error": self.1 });
        (self.0, Json(body)).into_response()
    }
}

fn bad_request(msg: impl ToString) -> AppError {
    AppError(StatusCode::BAD_REQUEST, msg.to_string())
}

fn internal(msg: impl ToString) -> AppError {
    AppError(StatusCode::INTERNAL_SERVER_ERROR, msg.to_string())
}

// -- Request / Response types -------------------------------------------------

#[derive(Deserialize)]
struct CreateCollectionReq {
    name: String,
    dim: usize,
    metric: Option<String>,
    m: Option<usize>,
    ef_construction: Option<usize>,
    quantize: Option<bool>,
}

#[derive(Deserialize)]
struct AddReq {
    id: String,
    vector: Vec<f32>,
    metadata: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct AddManyReq {
    ids: Vec<String>,
    vectors: Vec<Vec<f32>>,
    metadatas: Option<Vec<Option<serde_json::Value>>>,
}

#[derive(Deserialize)]
struct SearchReq {
    vector: Vec<f32>,
    k: Option<usize>,
    ef_search: Option<usize>,
    filter: Option<serde_json::Value>,
    max_distance: Option<f32>,
}

#[derive(Serialize)]
struct SearchResultResp {
    id: String,
    distance: f32,
    metadata: Option<serde_json::Value>,
}

// -- Handlers -----------------------------------------------------------------

async fn create_collection(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateCollectionReq>,
) -> AppResult<Json<serde_json::Value>> {
    let metric = parse_metric(req.metric.as_deref().unwrap_or("cosine"))
        .map_err(bad_request)?;
    let config = HnswConfig {
        m: req.m.unwrap_or(16),
        ef_construction: req.ef_construction.unwrap_or(200),
        quantize: req.quantize.unwrap_or(false),
    };

    let db = state.client.create_collection_with_config(&req.name, req.dim, metric, config)
        .map_err(bad_request)?;
    let db = Arc::new(db);
    state.collections.write().insert(req.name.clone(), db);

    Ok(Json(serde_json::json!({ "created": req.name })))
}

async fn list_collections(
    State(state): State<Arc<AppState>>,
) -> AppResult<Json<Vec<String>>> {
    state.client.list_collections().map(Json).map_err(|e| internal(e))
}

async fn delete_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> AppResult<Json<serde_json::Value>> {
    state.collections.write().remove(&name);
    let deleted = state.client.delete_collection(&name).map_err(|e| internal(e))?;
    Ok(Json(serde_json::json!({ "deleted": deleted })))
}

async fn add_vector(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<AddReq>,
) -> AppResult<Json<serde_json::Value>> {
    let db = state.get_or_open(&name)?;
    db.add(&req.id, req.vector, req.metadata).map_err(bad_request)?;
    Ok(Json(serde_json::json!({ "added": req.id })))
}

async fn upsert_vector(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<AddReq>,
) -> AppResult<Json<serde_json::Value>> {
    let db = state.get_or_open(&name)?;
    db.upsert(&req.id, req.vector, req.metadata).map_err(bad_request)?;
    Ok(Json(serde_json::json!({ "upserted": req.id })))
}

async fn add_many(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<AddManyReq>,
) -> AppResult<Json<serde_json::Value>> {
    let db = state.get_or_open(&name)?;
    let metas = req.metadatas.unwrap_or_else(|| vec![None; req.ids.len()]);
    let items: Vec<_> = req.ids.into_iter()
        .zip(req.vectors)
        .zip(metas)
        .map(|((id, vec), meta)| (id, vec, meta))
        .collect();
    let count = items.len();
    db.add_many(items).map_err(bad_request)?;
    Ok(Json(serde_json::json!({ "added": count })))
}

async fn search_vectors(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<SearchReq>,
) -> AppResult<Json<Vec<SearchResultResp>>> {
    let db = state.get_or_open(&name)?;
    let k = req.k.unwrap_or(10);
    let filter = req.filter.as_ref().map(parse_json_filter).transpose().map_err(bad_request)?;

    let results = db.search(&req.vector, k, req.ef_search, filter.as_ref(), req.max_distance)
        .map_err(bad_request)?;

    Ok(Json(results.into_iter().map(|r| SearchResultResp {
        id: r.id,
        distance: r.distance,
        metadata: r.metadata,
    }).collect()))
}

async fn get_vector(
    State(state): State<Arc<AppState>>,
    Path((name, id)): Path<(String, String)>,
) -> AppResult<Json<serde_json::Value>> {
    let db = state.get_or_open(&name)?;
    let (vector, metadata) = db.get(&id)
        .map_err(|e| AppError(StatusCode::NOT_FOUND, e.to_string()))?;
    Ok(Json(serde_json::json!({
        "id": id,
        "vector": vector,
        "metadata": metadata,
    })))
}

async fn delete_vector(
    State(state): State<Arc<AppState>>,
    Path((name, id)): Path<(String, String)>,
) -> AppResult<Json<serde_json::Value>> {
    let db = state.get_or_open(&name)?;
    let deleted = db.delete(&id).map_err(|e| internal(e))?;
    Ok(Json(serde_json::json!({ "deleted": deleted })))
}

async fn save_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> AppResult<Json<serde_json::Value>> {
    let db = state.get_or_open(&name)?;
    db.save().map_err(|e| internal(e))?;
    Ok(Json(serde_json::json!({ "saved": name })))
}

async fn collection_stats(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> AppResult<Json<serde_json::Value>> {
    let db = state.get_or_open(&name)?;
    let s = db.stats();
    Ok(Json(serde_json::json!({
        "num_vectors": s.num_vectors,
        "num_deleted": s.num_deleted,
        "num_layers": s.num_layers,
        "dim": db.dim(),
        "metric": format!("{:?}", db.metric()).to_lowercase(),
    })))
}

// -- Filter parsing -----------------------------------------------------------

fn parse_metric(s: &str) -> Result<Metric, String> {
    Metric::from_str(s).map_err(|e| e.to_string())
}

fn parse_json_filter(value: &serde_json::Value) -> Result<Filter, String> {
    vctrs_core::filter::parse_json_filter(value)
        .map_err(|e| e.to_string())
}

// -- Main ---------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let client = Client::new(&cli.data).expect("failed to open data directory");

    // Pre-load existing collections.
    let mut collections = HashMap::new();
    if let Ok(names) = client.list_collections() {
        for name in &names {
            if let Ok(db) = client.get_collection(name) {
                collections.insert(name.clone(), Arc::new(db));
            }
        }
        if !names.is_empty() {
            eprintln!("Loaded {} collection(s): {}", names.len(), names.join(", "));
        }
    }

    let state = Arc::new(AppState {
        client,
        collections: RwLock::new(collections),
    });

    let app = Router::new()
        .route("/collections", post(create_collection))
        .route("/collections", get(list_collections))
        .route("/collections/{name}", delete(delete_collection))
        .route("/collections/{name}/add", post(add_vector))
        .route("/collections/{name}/upsert", post(upsert_vector))
        .route("/collections/{name}/add_many", post(add_many))
        .route("/collections/{name}/search", post(search_vectors))
        .route("/collections/{name}/get/{id}", get(get_vector))
        .route("/collections/{name}/delete/{id}", delete(delete_vector))
        .route("/collections/{name}/save", post(save_collection))
        .route("/collections/{name}/stats", get(collection_stats))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", cli.host, cli.port);
    eprintln!("vctrs-server listening on http://{}", addr);
    eprintln!("Data directory: {}", cli.data);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
