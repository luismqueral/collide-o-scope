#include "ControlPanel.h"

#include <QApplication>
#include <QUrl>

// Entry point: spin up the Qt event loop and show the control panel pointed at
// the engine's WebSocket. The URL matches the Rust server in src/web/server.rs
// (127.0.0.1:3030, route "/ws"). Launch order doesn't matter — EngineClient
// auto-reconnects, so this can start before `cargo run -- library/`.
int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    ControlPanel panel(QUrl(QStringLiteral("ws://127.0.0.1:3030/ws")));
    panel.setWindowTitle(QStringLiteral("collide-o-scope \u00B7 Qt panel"));
    panel.resize(560, 720);
    panel.show();

    return app.exec();
}
