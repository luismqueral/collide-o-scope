#pragma once

#include <QAbstractSocket>
#include <QJsonObject>
#include <QObject>
#include <QUrl>
#include <QtWebSockets/QWebSocket>

class QTimer;

// EngineClient is the C++ mirror of the browser<->engine wire contract defined
// in the Rust side at src/web/state.rs. It owns one WebSocket to the render
// engine and:
//
//   * keeps it connected, auto-reconnecting on a timer if the engine isn't up
//     yet or restarts (so launch order doesn't matter);
//   * turns every inbound AppSnapshot JSON object into a snapshotReceived()
//     signal (the engine pushes one on connect, then ~30x/sec);
//   * sends WebAction JSON back out via sendAction() + small typed helpers.
//
// Everything above the socket is plain Qt signals/slots, so the UI never
// touches the network directly.
class EngineClient : public QObject {
    Q_OBJECT
public:
    explicit EngineClient(QUrl url, QObject *parent = nullptr);

    bool isConnected() const { return m_connected; }

    // --- outbound WebActions (tags/fields match web/state.rs exactly) ---
    void sendAction(const QJsonObject &action);
    void setParam(const QString &param, double value);          // set_param (master FX)
    void setParam(const QString &param, bool value);            // set_param (e.g. invert)
    void setMasterAudioParam(const QString &param, double value);
    void setMasterAudioParam(const QString &param, bool value); // e.g. limiter
    void setLayerParam(int index, const QString &param, double value);
    void resetFx();                                             // reset_fx
    void toggleMasterPause();                                   // toggle_master_pause
    void toggleVisibility(int index);                           // toggle_visibility
    void toggleLayerPause(int index);                           // toggle_layer_pause
    void focusWindow();                                         // focus_window

signals:
    void snapshotReceived(const QJsonObject &snapshot);
    void connectionChanged(bool connected);

private slots:
    void onConnected();
    void onDisconnected();
    void onTextMessage(const QString &message);
    void tryReconnect();

private:
    QUrl m_url;
    QWebSocket m_socket;
    QTimer *m_reconnectTimer = nullptr;
    bool m_connected = false;
};
