//! Axum HTTP + WebSocket server for the control panel.

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

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime");

    let addr_clone = addr.clone();
    std::thread::spawn(move || {
        rt.block_on(async move {
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

async fn thumb_handler(
    Path(filename): Path<String>,
    State(state): State<Arc<WebState>>,
) -> impl IntoResponse {
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

async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<WebState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: Arc<WebState>) {
    let (mut sender, mut receiver) = socket.split();

    // Send current state on connect
    let current = state.app.read().await;
    let init_msg = serde_json::to_string(&*current).unwrap();
    drop(current);
    let _ = sender.send(Message::Text(init_msg.into())).await;

    // Subscribe to broadcast updates (state JSON)
    let mut rx = state.tx.subscribe();

    // Forward broadcasts to this client
    let mut send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if sender.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    // Receive actions from this client
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

    tokio::select! {
        _ = &mut send_task => recv_task.abort(),
        _ = &mut recv_task => send_task.abort(),
    }
}
