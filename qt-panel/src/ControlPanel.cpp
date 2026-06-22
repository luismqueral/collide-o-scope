#include "ControlPanel.h"

#include "EngineClient.h"
#include "MatrixCell.h"
#include "Schema.h"

#include <QFontMetrics>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QJsonArray>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QVBoxLayout>

#include <utility>

namespace {
// Progress/meter bars run at this integer resolution (values are 0..1 floats).
constexpr int kBars = 1000;

// Shared label styling so the Qt panel matches the browser's dark theme.
QLabel *groupHeader(const QString &text) {
    auto *l = new QLabel(text);
    l->setStyleSheet(QStringLiteral(
        "color:#7a8aa0; font-weight:700; font-size:10px; padding-top:7px; letter-spacing:1px;"));
    return l;
}
QLabel *rowLabel(const QString &text) {
    auto *l = new QLabel(text);
    l->setStyleSheet(QStringLiteral("color:#9a9aa8; font-size:11px;"));
    return l;
}

const char *kProgressQss =
    "QProgressBar{background:#1c1c24;border:none;} QProgressBar::chunk{background:#5a8fe6;}";
const char *kMeterQss =
    "QProgressBar{background:#1c1c24;border:none;} QProgressBar::chunk{background:#3ad15a;}";
} // namespace

ControlPanel::ControlPanel(QUrl engineUrl, QWidget *parent) : QWidget(parent) {
    m_engine = new EngineClient(std::move(engineUrl), this);
    connect(m_engine, &EngineClient::snapshotReceived, this, &ControlPanel::onSnapshot);
    connect(m_engine, &EngineClient::connectionChanged, this, &ControlPanel::onConnectionChanged);

    // Global dark theme (cells paint their own backgrounds; this covers the rest).
    setStyleSheet(QStringLiteral(
        "QWidget{background:#0f0f12;color:#d8d8dc;}"
        "QScrollArea{border:1px solid #2a2a35;}"
        "QGroupBox{border:1px solid #2a2a35;border-radius:4px;margin-top:8px;}"
        "QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 3px;color:#9a9aa8;}"
        "QPushButton{background:#1c1c24;border:1px solid #2a2a35;border-radius:3px;padding:3px "
        "8px;}"
        "QPushButton:hover{border-color:#5a8fe6;}"
        "QPushButton:checked{background:#5a8fe6;color:#0f0f12;border-color:#5a8fe6;}"));

    auto *root = new QVBoxLayout(this);
    root->addWidget(buildConnectionBar());
    root->addWidget(buildTransport());

    auto *body = new QHBoxLayout;
    body->addWidget(buildLayerPanel(), 1); // the layer grid soaks up extra width
    body->addWidget(buildMasterPanel());
    root->addLayout(body, 1);

    onConnectionChanged(false); // paint the initial "reconnecting" state
}

QWidget *ControlPanel::buildConnectionBar() {
    auto *box = new QWidget;
    auto *h = new QHBoxLayout(box);
    h->setContentsMargins(0, 0, 0, 0);
    m_statusDot = new QLabel(QStringLiteral("\u25CF")); // ●
    m_statusText = new QLabel;
    h->addWidget(m_statusDot);
    h->addWidget(m_statusText);
    h->addStretch();
    return box;
}

QWidget *ControlPanel::buildTransport() {
    auto *box = new QGroupBox(QStringLiteral("Master"));
    auto *h = new QHBoxLayout(box);
    m_pauseBtn = new QPushButton(QStringLiteral("Pause"));
    connect(m_pauseBtn, &QPushButton::clicked, this, [this] { m_engine->toggleMasterPause(); });
    auto *focusBtn = new QPushButton(QStringLiteral("Focus output window"));
    connect(focusBtn, &QPushButton::clicked, this, [this] { m_engine->focusWindow(); });
    h->addWidget(m_pauseBtn);
    h->addWidget(focusBtn);
    h->addStretch();
    return box;
}

QWidget *ControlPanel::buildLayerPanel() {
    m_layerScroll = new QScrollArea;
    m_layerScroll->setWidgetResizable(true);
    rebuildLayerGrid(0); // empty grid (labels + "+" only) until the first snapshot
    return m_layerScroll;
}

QWidget *ControlPanel::buildMasterPanel() {
    auto *scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setMinimumWidth(220);
    scroll->setMaximumWidth(300);

    auto *inner = new QWidget;
    auto *grid = new QGridLayout(inner);
    grid->setHorizontalSpacing(6);
    grid->setVerticalSpacing(2);

    int row = 0;
    grid->addWidget(groupHeader(QStringLiteral("MASTER")), row++, 0, 1, 2);
    for (const Group &g : matrixGroups()) {
        // Only the rows that apply to the master column.
        QVector<const ParamDef *> ps;
        for (const ParamDef &p : g.params)
            if (channelApplies(p, ColumnKind::Master))
                ps.push_back(&p);
        if (ps.isEmpty())
            continue;

        grid->addWidget(groupHeader(g.name), row++, 0, 1, 2);
        for (const ParamDef *def : ps) {
            grid->addWidget(rowLabel(def->label), row, 0);
            auto *cell = new MatrixCell(def, ColumnDesc{ColumnKind::Master, 0}, m_engine, nullptr);
            cell->setMinimumWidth(120);
            grid->addWidget(cell, row, 1);
            m_masterCells.push_back(cell);
            ++row;
        }
        // The master output meter sits under the AUDIO group (read-only peak).
        if (g.name == QStringLiteral("AUDIO")) {
            grid->addWidget(rowLabel(QStringLiteral("output")), row, 0);
            m_masterMeter = new QProgressBar;
            m_masterMeter->setRange(0, kBars);
            m_masterMeter->setTextVisible(false);
            m_masterMeter->setFixedHeight(12);
            m_masterMeter->setStyleSheet(QString::fromLatin1(kMeterQss));
            grid->addWidget(m_masterMeter, row, 1);
            ++row;
        }
    }
    grid->setColumnStretch(1, 1);
    grid->setRowStretch(row, 1);

    scroll->setWidget(inner);
    return scroll;
}

void ControlPanel::rebuildLayerGrid(int count) {
    m_layerCells.clear();
    m_layerHeaders.clear();

    // Fresh inner widget + grid each time (setWidget deletes the previous one,
    // taking all its child cells/headers with it — so no manual cleanup needed).
    m_layerInner = new QWidget;
    m_layerGrid = new QGridLayout(m_layerInner);
    m_layerGrid->setHorizontalSpacing(2);
    m_layerGrid->setVerticalSpacing(2);

    // Row 0: column headers. Col 0 corner, then a header per layer, then "+".
    m_layerGrid->addWidget(groupHeader(QStringLiteral("LAYERS")), 0, 0);
    for (int i = 0; i < count; ++i)
        m_layerGrid->addWidget(makeLayerHeader(i), 0, i + 1);

    m_addColBtn = new QPushButton(QStringLiteral("+"));
    m_addColBtn->setFixedWidth(28);
    connect(m_addColBtn, &QPushButton::clicked, this, [this] { addLayerViaPicker(); });
    m_layerGrid->addWidget(m_addColBtn, 0, count + 1, Qt::AlignTop);

    // Param rows, grouped per LAYER_GROUPS. Keys resolve to shared ParamDefs.
    int row = 1;
    for (const LayerGroup &g : layerGroups()) {
        m_layerGrid->addWidget(groupHeader(g.name), row++, 0);
        for (const QString &key : g.keys) {
            const ParamDef *def = findParam(key);
            if (!def)
                continue;
            m_layerGrid->addWidget(rowLabel(def->label), row, 0);
            for (int i = 0; i < count; ++i) {
                auto *cell =
                    new MatrixCell(def, ColumnDesc{ColumnKind::Layer, i}, m_engine, &m_library);
                cell->setMinimumWidth(70);
                m_layerGrid->addWidget(cell, row, i + 1);
                m_layerCells.push_back(cell);
            }
            ++row;
        }
    }
    m_layerGrid->setColumnStretch(count + 1, 1); // keep columns packed left
    m_layerGrid->setRowStretch(row, 1);

    m_layerScroll->setWidget(m_layerInner);
    m_layerCount = count;
}

QWidget *ControlPanel::makeLayerHeader(int index) {
    LayerHeader h;
    h.container = new QWidget;
    auto *v = new QVBoxLayout(h.container);
    v->setContentsMargins(2, 2, 2, 2);
    v->setSpacing(1);

    auto *top = new QHBoxLayout;
    top->setSpacing(2);
    h.title = new QLabel(QStringLiteral("L%1").arg(index + 1));
    h.title->setStyleSheet(QStringLiteral("color:#5a8fe6; font-weight:700;"));
    auto *rm = new QPushButton(QStringLiteral("\u00D7")); // ×
    rm->setFixedSize(16, 16);
    rm->setStyleSheet(QStringLiteral("padding:0; color:#9a9aa8;"));
    connect(rm, &QPushButton::clicked, this, [this, index] { m_engine->removeLayer(index); });
    top->addWidget(h.title);
    top->addStretch();
    top->addWidget(rm);
    v->addLayout(top);

    h.filename = new QLabel;
    h.filename->setStyleSheet(QStringLiteral("color:#6a6a78; font-size:10px;"));
    h.filename->setMaximumWidth(84);
    v->addWidget(h.filename);

    h.progress = new QProgressBar;
    h.progress->setRange(0, kBars);
    h.progress->setTextVisible(false);
    h.progress->setFixedHeight(4);
    h.progress->setStyleSheet(QString::fromLatin1(kProgressQss));
    v->addWidget(h.progress);

    h.meter = new QProgressBar;
    h.meter->setRange(0, kBars);
    h.meter->setTextVisible(false);
    h.meter->setFixedHeight(4);
    h.meter->setStyleSheet(QString::fromLatin1(kMeterQss));
    v->addWidget(h.meter);

    m_layerHeaders.push_back(h);
    return h.container;
}

void ControlPanel::addLayerViaPicker() {
    if (m_library.isEmpty())
        return;
    bool ok = false;
    const QString choice = QInputDialog::getItem(this, QStringLiteral("Add layer"),
                                                 QStringLiteral("Clip:"), m_library, 0, false, &ok);
    if (ok && !choice.isEmpty())
        m_engine->addLayer(choice);
}

void ControlPanel::onConnectionChanged(bool connected) {
    if (connected) {
        m_statusDot->setStyleSheet(QStringLiteral("color: #3ad15a;"));
        m_statusText->setText(QStringLiteral("Connected — ws://127.0.0.1:3030/ws"));
    } else {
        m_statusDot->setStyleSheet(QStringLiteral("color: #e0a23a;"));
        m_statusText->setText(QStringLiteral("Reconnecting to ws://127.0.0.1:3030/ws …"));
    }
}

void ControlPanel::onSnapshot(const QJsonObject &snap) {
    // Refresh the library list first so the add/swap pickers are current.
    m_library.clear();
    const QJsonArray lib = snap.value(QStringLiteral("library")).toArray();
    for (const QJsonValue &v : lib)
        m_library.push_back(v.toString());

    // Master transport.
    const bool paused = snap.value(QStringLiteral("paused")).toBool();
    m_pauseBtn->setText(paused ? QStringLiteral("Resume") : QStringLiteral("Pause"));

    // Rebuild the layer grid only when the column count changes.
    const QJsonArray layers = snap.value(QStringLiteral("layers")).toArray();
    if (layers.size() != m_layerCount)
        rebuildLayerGrid(static_cast<int>(layers.size()));

    // Fan the snapshot out to every cell (each self-guards while scrubbing).
    for (MatrixCell *c : m_masterCells)
        c->applyValue(snap);
    for (MatrixCell *c : m_layerCells)
        c->applyValue(snap);

    if (m_masterMeter)
        m_masterMeter->setValue(
            static_cast<int>(snap.value(QStringLiteral("meter")).toDouble() * kBars));

    // Live per-column headers: filename (♪ for audio-only) + progress + meter.
    for (int i = 0; i < layers.size() && i < m_layerHeaders.size(); ++i) {
        const QJsonObject l = layers.at(i).toObject();
        const LayerHeader &h = m_layerHeaders.at(i);
        const QString fn = l.value(QStringLiteral("filename")).toString();
        const bool audioOnly = l.value(QStringLiteral("audio_only")).toBool();
        const QString shown = (audioOnly ? QString::fromUtf8("\u266A ") : QString()) + fn;
        const QFontMetrics fm(h.filename->font());
        h.filename->setText(fm.elidedText(shown, Qt::ElideRight, h.filename->maximumWidth()));
        h.filename->setToolTip(fn);
        h.progress->setValue(
            static_cast<int>(l.value(QStringLiteral("progress")).toDouble() * kBars));
        h.meter->setValue(static_cast<int>(l.value(QStringLiteral("meter")).toDouble() * kBars));
    }
}
