//! PegaFlow P/D Disaggregation Router
//!
//! Simple router that coordinates prefill (P) and decode (D) nodes.
//! Flow:
//! 1. Receive request
//! 2. Send to P node (max_tokens=1)
//! 3. Forward to D node (P response means KV is ready)

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Instant;

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use clap::Parser;
use reqwest::Client;
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio_stream::StreamExt;
use tracing::{error, info};

#[derive(Clone)]
struct RouterState {
    prefill_clients: Arc<Vec<Client>>,
    decode_clients: Arc<Vec<Client>>,
    prefill_urls: Arc<Vec<String>>,
    decode_urls: Arc<Vec<String>>,
    p_index: Arc<AtomicUsize>,
    d_index: Arc<AtomicUsize>,
}

impl RouterState {
    fn new(prefill_endpoints: Vec<String>, decode_endpoints: Vec<String>) -> Self {
        let prefill_clients = prefill_endpoints
            .iter()
            .map(|_| {
                Client::builder()
                    .build()
                    .expect("Failed to build prefill client")
            })
            .collect();

        let decode_clients = decode_endpoints
            .iter()
            .map(|_| {
                Client::builder()
                    .build()
                    .expect("Failed to build decode client")
            })
            .collect();

        Self {
            prefill_clients: Arc::new(prefill_clients),
            decode_clients: Arc::new(decode_clients),
            prefill_urls: Arc::new(prefill_endpoints),
            decode_urls: Arc::new(decode_endpoints),
            p_index: Arc::new(AtomicUsize::new(0)),
            d_index: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn get_next_p(&self) -> (Client, String) {
        let idx = self.p_index.fetch_add(1, Ordering::Relaxed);
        let idx = idx % self.prefill_clients.len();
        (
            self.prefill_clients[idx].clone(),
            self.prefill_urls[idx].clone(),
        )
    }

    fn get_next_d(&self) -> (Client, String) {
        let idx = self.d_index.fetch_add(1, Ordering::Relaxed);
        let idx = idx % self.decode_clients.len();
        (
            self.decode_clients[idx].clone(),
            self.decode_urls[idx].clone(),
        )
    }
}

async fn handle_completion(
    State(state): State<RouterState>,
    _headers: HeaderMap,
    Json(body): Json<Value>,
    api_path: &str,
) -> Response {
    let arrive_time = Instant::now();

    // Use existing request_id or generate new one
    let req_id = body
        .get("request_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    info!("request arrived: req={}", req_id);

    // Save original values to restore for D request
    let org_max_tokens = body.get("max_tokens").cloned();
    let org_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let stream_options = body.get("stream_options").cloned();

    // Prepare P request (max_tokens=1, non-streaming)
    let mut p_body = body.clone();
    p_body["max_tokens"] = json!(1);
    p_body["stream"] = json!(false);
    p_body["request_id"] = json!(req_id.clone());

    // Remove stream_options since stream=false
    p_body.as_object_mut().map(|obj| obj.remove("stream_options"));

    // Ensure min_tokens <= max_tokens to avoid 400 from P node
    if let Some(min_tokens) = p_body.get("min_tokens").and_then(|v| v.as_i64()) {
        p_body["min_tokens"] = json!(min_tokens.min(1));
    } else {
        p_body["min_tokens"] = json!(0);
    }

    let (p_client, p_url) = state.get_next_p();
    let p_url = format!("{}{}", p_url, api_path);

    // Send to P node
    let p_response = match p_client
        .post(&p_url)
        .header("X-Request-Id", &req_id)
        .json(&p_body)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            error!("P request failed: req={} error={}", req_id, e);
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": format!("Prefill node error: {}", e)})),
            )
                .into_response();
        }
    };

    let p_status = p_response.status();
    let p_result: Value = match p_response.json().await {
        Ok(v) => v,
        Err(e) => {
            error!("P response parse failed: req={} error={}", req_id, e);
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": format!("Prefill response error: {}", e)})),
            )
                .into_response();
        }
    };

    if !p_status.is_success() {
        error!(
            "P error: req={} status={} body={:?}",
            req_id, p_status, p_result
        );
        return (p_status, Json(p_result)).into_response();
    }

    let prefill_latency = arrive_time.elapsed().as_millis();
    info!("prefill done: req={} latency={}ms", req_id, prefill_latency);

    // Prepare D request (restore original settings)
    let mut d_body = body;
    if let Some(max_tokens) = org_max_tokens {
        d_body["max_tokens"] = max_tokens;
    }
    d_body["stream"] = json!(org_stream);
    d_body["request_id"] = json!(req_id.clone());
    if let Some(stream_opts) = stream_options {
        d_body["stream_options"] = stream_opts;
    }

    let (d_client, d_url) = state.get_next_d();
    let d_url = format!("{}{}", d_url, api_path);

    if org_stream {
        // Streaming response
        let d_response = match d_client
            .post(&d_url)
            .header("X-Request-Id", &req_id)
            .json(&d_body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                error!("D request failed: req={} error={}", req_id, e);
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": format!("Decode node error: {}", e)})),
                )
                    .into_response();
            }
        };

        let stream = d_response.bytes_stream();
        let stream = tokio_stream::wrappers::ReceiverStream::new({
            let (tx, rx) = tokio::sync::mpsc::channel(100);
            let req_id = req_id.clone();
            tokio::spawn(async move {
                let mut stream = Box::pin(stream);
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            if tx.send(Ok::<_, std::io::Error>(bytes)).await.is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            error!("Stream error: req={} error={}", req_id, e);
                            break;
                        }
                    }
                }
                let total = arrive_time.elapsed().as_millis();
                info!("done (stream): req={} total={}ms", req_id, total);
            });
            rx
        });

        Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/event-stream")
            .body(Body::from_stream(stream))
            .unwrap()
    } else {
        // Non-streaming response
        let d_response = match d_client
            .post(&d_url)
            .header("X-Request-Id", &req_id)
            .json(&d_body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                error!("D request failed: req={} error={}", req_id, e);
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": format!("Decode node error: {}", e)})),
                )
                    .into_response();
            }
        };

        let d_status = d_response.status();
        let d_result: Value = match d_response.json().await {
            Ok(v) => v,
            Err(e) => {
                error!("D response parse failed: req={} error={}", req_id, e);
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({"error": format!("Decode response error: {}", e)})),
                )
                    .into_response();
            }
        };

        let total = arrive_time.elapsed().as_millis();
        info!("done: req={} total={}ms", req_id, total);

        (d_status, Json(d_result)).into_response()
    }
}

async fn chat_completions(
    state: State<RouterState>,
    headers: HeaderMap,
    body: Json<Value>,
) -> Response {
    handle_completion(state, headers, body, "/v1/chat/completions").await
}

async fn completions(state: State<RouterState>, headers: HeaderMap, body: Json<Value>) -> Response {
    handle_completion(state, headers, body, "/v1/completions").await
}

#[derive(Parser)]
#[command(name = "pegaflow-router")]
#[command(about = "PegaFlow P/D Disaggregation Router")]
struct Args {
    /// Host to bind
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to bind
    #[arg(long, default_value = "8080")]
    port: u16,

    /// Prefill endpoints
    #[arg(long, required = true, num_args = 1..)]
    prefill: Vec<String>,

    /// Decode endpoints
    #[arg(long, required = true, num_args = 1..)]
    decode: Vec<String>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    let state = RouterState::new(args.prefill.clone(), args.decode.clone());

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state);

    let addr = format!("{}:{}", args.host, args.port);
    info!("Starting on {}", addr);
    info!("Prefill nodes: {:?}", args.prefill);
    info!("Decode nodes: {:?}", args.decode);

    let listener = TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
