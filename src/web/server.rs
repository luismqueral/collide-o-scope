//! Axum HTTP + WebSocket server for the control panel.
//!
//! Runs entirely on its own background thread with its own tokio async runtime,
//! so the GPU render loop on the main thread is never blocked by web I/O. The
//! two sides share one `Arc<WebState>` (a thread-safe reference-counted handle):
//! the browser pushes `WebAction`s into `state.actions`, and the render loop
//! broadcasts JSON snapshots back out through `state.tx`.
//!
//! Routes: `/ws` (the live control channel), `/thumb/...` and `/preview/...`
//! (cached JPEGs for the library browser), and a fallback that serves the
//! embedded static panel files.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Path, State, WebSocketUpgrade};
use axum::http::{header, StatusCode};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use futures::{SinkExt, StreamExt};

use super::state::{WebAction, WebState};
use super::static_files;

/// Start the web server on a background thread. Returns the URL.
pub fn spawn(state: Arc<WebState>, port: u16) -> String {
    let addr = format!("127.0.0.1:{port}");
    let url = format!("http://{addr}");

    // tokio is an async runtime — it drives all the `async fn`s below. We build
    // one explicitly (rather than the `#[tokio::main]` macro) because the main
    // thread belongs to winit/wgpu, so the runtime has to live on its own thread.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime");

    // `clone()` here bumps the address String / Arc refcount so the spawned
    // closure can `move`-capture its own owned copy (the original `addr` is
    // still used afterwards to build `url`). The new OS thread runs the server
    // forever via `block_on`, parking this thread inside the async runtime.
    let addr_clone = addr.clone();
    std::thread::spawn(move || {
        rt.block_on(async move {
            // Axum router: map URL paths to handler functions. `.fallback`
            // catches anything unmatched and serves a static file. `with_state`
            // makes the shared `WebState` injectable into every handler.
            let app = Router::new()
                .route("/ws", get(ws_handler))
                .route("/thumb/:filename", get(thumb_handler))
                .route("/preview/:filename/:index", get(preview_handler))
                .fallback(get(static_files::serve))
                .with_state(state);

            let listener = tokio::net::TcpListener::bind(&addr_clone)
                .await
                .expect("Failed to bind web server");

            log::info!("Web control panel: http://{}", addr_clone);

            axum::serve(listener, app).await.unwrap();
        });
    });

    url
}

/// Serve a cached library thumbnail JPEG by filename. The `Path(...)` and
/// `State(...)` parameters are axum "extractors" — axum fills them in from the
/// URL segment and the shared state respectively before calling this function.
async fn thumb_handler(
    Path(filename): Path<String>,
    State(state): State<Arc<WebState>>,
) -> impl IntoResponse {
    // `.read()` takes a read lock on the cache; `if let Ok(...)` skips the body
    // if the lock is poisoned. We `clone()` the bytes so the lock is released
    // as soon as this function returns.
    if let Ok(cache) = state.thumbnails.read() {
        if let Some(jpeg) = cache.get(&filename) {
            return (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "image/jpeg")],
                jpeg.clone(),
            )
                .into_response();
        }
    }
    StatusCode::NOT_FOUND.into_response()
}

/// Serve one frame of a hover-scrub preview: `/preview/<file>/<index>` returns
/// the Nth cached JPEG for that clip.
async fn preview_handler(
    Path((filename, index)): Path<(String, usize)>,
    State(state): State<Arc<WebState>>,
) -> impl IntoResponse {
    if let Ok(cache) = state.preview_frames.read() {
        if let Some(frames) = cache.get(&filename) {
            if let Some(jpeg) = frames.get(index) {
                return (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "image/jpeg")],
                    jpeg.clone(),
                )
                    .into_response();
            }
        }
    }
    StatusCode::NOT_FOUND.into_response()
}

/// HTTP→WebSocket upgrade entry point. The browser opens `ws://.../ws`; axum
/// performs the protocol handshake and then hands us the live socket.
async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<WebState>>) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// Drive one connected browser client: stream snapshots out, read actions in.
async fn handle_socket(socket: WebSocket, state: Arc<WebState>) {
    // Split the socket so the send and receive halves can be owned by two
    // independent tasks running concurrently.
    let (mut sender, mut receiver) = socket.split();

    // Send current state on connect so the UI paints immediately instead of
    // waiting for the next broadcast tick. `drop(current)` releases the read
    // lock before we await the (potentially slow) network send.
    let current = state.app.read().await;
    let init_msg = serde_json::to_string(&*current).unwrap();
    drop(current);
    let _ = sender.send(Message::Text(init_msg.into())).await;

    // `subscribe()` to the broadcast channel: every snapshot the render loop
    // publishes will arrive on `rx`, fanned out to all connected clients.
    let mut rx = state.tx.subscribe();

    // Task 1: forward every broadcast snapshot to this client. Breaks out (and
    // ends the task) once the socket send fails, i.e. the client disconnected.
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if sender.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    // Task 2: read inbound messages, parse each as a `WebAction`, and push it
    // onto the shared queue for the render loop to drain. Unparseable messages
    // are logged and skipped rather than killing the connection.
    let state_clone = state.clone();
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            if let Message::Text(text) = msg {
                // Try to parse as a WebAction
                match serde_json::from_str::<WebAction>(&text) {
                    Ok(action) => {
                        state_clone.actions.lock().await.push(action);
                    }
                    Err(e) => {
                        log::warn!("Failed to parse WebAction: {e} — raw: {text}");
                    }
                }
            }
        }
    });

    // Wait for *either* task to finish (whichever happens first on disconnect),
    // then abort the other so we don't leak a dangling task per closed socket.
    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    }
}
