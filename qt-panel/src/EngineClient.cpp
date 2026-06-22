#include "EngineClient.h"

#include <QJsonDocument>
#include <QTimer>
#include <utility>

EngineClient::EngineClient(QUrl url, QObject *parent)
    : QObject(parent), m_url(std::move(url)) {
    // Retry timer first, so the signal handlers below can safely reference it.
    m_reconnectTimer = new QTimer(this);
    m_reconnectTimer->setInterval(1000);
    connect(m_reconnectTimer, &QTimer::timeout, this, &EngineClient::tryReconnect);

    connect(&m_socket, &QWebSocket::connected, this, &EngineClient::onConnected);
    connect(&m_socket, &QWebSocket::disconnected, this, &EngineClient::onDisconnected);
    connect(&m_socket, &QWebSocket::textMessageReceived, this, &EngineClient::onTextMessage);
    // A *failed* connect (engine not running yet) emits errorOccurred but not
    // disconnected, so arm the retry timer here too.
    connect(&m_socket, &QWebSocket::errorOccurred, this,
            [this](QAbstractSocket::SocketError) {
                if (!m_connected && !m_reconnectTimer->isActive())
                    m_reconnectTimer->start();
            });

    // First attempt immediately; the timer covers everything after.
    m_socket.open(m_url);
}

void EngineClient::onConnected() {
    m_connected = true;
    m_reconnectTimer->stop();
    emit connectionChanged(true);
}

void EngineClient::onDisconnected() {
    if (m_connected) {
        m_connected = false;
        emit connectionChanged(false);
    }
    if (!m_reconnectTimer->isActive())
        m_reconnectTimer->start();
}

void EngineClient::tryReconnect() {
    if (m_connected) {
        m_reconnectTimer->stop();
        return;
    }
    // Only re-open from a clean state; if a previous attempt is still pending
    // (ConnectingState) we let it run rather than stacking opens.
    if (m_socket.state() == QAbstractSocket::UnconnectedState)
        m_socket.open(m_url);
}

void EngineClient::onTextMessage(const QString &message) {
    const QJsonDocument doc = QJsonDocument::fromJson(message.toUtf8());
    if (doc.isObject())
        emit snapshotReceived(doc.object());
}

void EngineClient::sendAction(const QJsonObject &action) {
    if (!m_connected)
        return; // dropped silently until the socket is up — fine for a control surface
    m_socket.sendTextMessage(QString::fromUtf8(QJsonDocument(action).toJson(QJsonDocument::Compact)));
}

// --- typed convenience wrappers around sendAction ---
// Built field-by-field (rather than brace-init) to keep the JSON value types
// unambiguous and the wire shape obvious at a glance.

void EngineClient::setParam(const QString &param, double value) {
    QJsonObject a;
    a["action"] = "set_param";
    a["param"] = param;
    a["value"] = value;
    sendAction(a);
}

void EngineClient::setParam(const QString &param, bool value) {
    QJsonObject a;
    a["action"] = "set_param";
    a["param"] = param;
    a["value"] = value;
    sendAction(a);
}

void EngineClient::setMasterAudioParam(const QString &param, double value) {
    QJsonObject a;
    a["action"] = "set_master_audio_param";
    a["param"] = param;
    a["value"] = value;
    sendAction(a);
}

void EngineClient::setMasterAudioParam(const QString &param, bool value) {
    QJsonObject a;
    a["action"] = "set_master_audio_param";
    a["param"] = param;
    a["value"] = value;
    sendAction(a);
}

void EngineClient::setLayerParam(int index, const QString &param, double value) {
    QJsonObject a;
    a["action"] = "set_layer_param";
    a["index"] = index;
    a["param"] = param;
    a["value"] = value;
    sendAction(a);
}

void EngineClient::resetFx() {
    QJsonObject a;
    a["action"] = "reset_fx";
    sendAction(a);
}

void EngineClient::toggleMasterPause() {
    QJsonObject a;
    a["action"] = "toggle_master_pause";
    sendAction(a);
}

void EngineClient::toggleVisibility(int index) {
    QJsonObject a;
    a["action"] = "toggle_visibility";
    a["index"] = index;
    sendAction(a);
}

void EngineClient::toggleLayerPause(int index) {
    QJsonObject a;
    a["action"] = "toggle_layer_pause";
    a["index"] = index;
    sendAction(a);
}

void EngineClient::focusWindow() {
    QJsonObject a;
    a["action"] = "focus_window";
    sendAction(a);
}
