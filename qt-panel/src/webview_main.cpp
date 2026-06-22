// Native-window wrapper around the *browser* control panel.
//
// This is the second of the two panel experiments. Where collide-panel
// re-implements the UI natively (QPainter cells, lowest-latency meters), this
// one takes the opposite tack: it loads the engine's existing web UI
// (http://127.0.0.1:3030, served from src/web/static_files.rs) inside a
// Qt WebEngine (Chromium) view, so we get the *exact* "perfect" browser UI but
// in a chromeless, native-feeling window — no address bar, own dock icon.
//
// It is built and run completely separately from the Rust engine; like the
// native panel it is a pure client and makes ZERO engine changes. The two
// binaries exist side by side on purpose, to compare native-render vs.
// web-render latency (audio meters especially).
//
// Qt WebEngine is cross-platform (Windows/macOS/Linux) and ships its own
// Chromium, so the UI renders identically on every desktop platform.

#include <QApplication>
#include <QGuiApplication>
#include <QRect>
#include <QScreen>
#include <QTimer>
#include <QUrl>
#include <QWebEngineView>

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    // The engine passes the panel URL as the first argument (so it can point us
    // at e.g. "…?view=classic"); standalone launches fall back to the default.
    const QUrl panelUrl(argc > 1 ? QString::fromUtf8(argv[1])
                                 : QStringLiteral("http://127.0.0.1:3030"));

    QWebEngineView view;
    view.setWindowTitle(QStringLiteral("collide-o-scope \u00B7 web panel"));

    // Dock the panel to the left of the primary screen at full available height,
    // so it sits side-by-side with the engine's preview window (which the engine
    // places just to our right). The 0.58 split is mirrored in src/main.rs's
    // preview-window positioning — keep the two values in sync.
    if (QScreen *screen = QGuiApplication::primaryScreen()) {
        const QRect avail = screen->availableGeometry();
        const int panelW = int(avail.width() * 0.58);
        view.setGeometry(avail.x(), avail.y(), panelW, avail.height());
    } else {
        view.resize(1200, 800);
    }

    // Launch order shouldn't matter (matching the native panel's auto-reconnect
    // convenience). The HTTP GET fails if the engine isn't up yet, so on a failed
    // load we just retry after a short delay until the server answers. Once the
    // page loads, the panel's own JS owns the live WebSocket reconnection.
    QObject::connect(&view, &QWebEngineView::loadFinished, &view, [&view, panelUrl](bool ok) {
        if (!ok)
            QTimer::singleShot(1000, &view, [&view, panelUrl] { view.load(panelUrl); });
    });

    view.load(panelUrl);
    view.show();

    return app.exec();
}
