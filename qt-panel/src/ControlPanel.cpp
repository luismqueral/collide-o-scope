#include "ControlPanel.h"

#include "EngineClient.h"

#include <QCheckBox>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QLabel>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSlider>
#include <QVBoxLayout>

#include <algorithm>
#include <cmath>
#include <utility>

namespace {
// The integer resolution every QSlider runs at internally; we map each param's
// real float range onto 0..kSliderSteps.
constexpr int kSliderSteps = 1000;

QString fmt(double v) { return QString::number(v, 'f', 2); }

int floatToSlider(double v, double mn, double mx) {
    const double t = (v - mn) / (mx - mn);
    return std::clamp(static_cast<int>(std::lround(t * kSliderSteps)), 0, kSliderSteps);
}

double sliderToFloat(int s, double mn, double mx) {
    return mn + (mx - mn) * (static_cast<double>(s) / kSliderSteps);
}
} // namespace

ControlPanel::ControlPanel(QUrl engineUrl, QWidget *parent) : QWidget(parent) {
    m_engine = new EngineClient(std::move(engineUrl), this);
    connect(m_engine, &EngineClient::snapshotReceived, this, &ControlPanel::onSnapshot);
    connect(m_engine, &EngineClient::connectionChanged, this, &ControlPanel::onConnectionChanged);

    auto *root = new QVBoxLayout(this);
    root->addWidget(buildConnectionBar());
    root->addWidget(buildTransport());
    root->addWidget(buildMasterFx());
    root->addWidget(buildMasterAudio());
    root->addWidget(buildLayers(), 1); // the layer strip soaks up extra height

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

QWidget *ControlPanel::buildMasterFx() {
    auto *box = new QGroupBox(QStringLiteral("Master FX"));
    auto *v = new QVBoxLayout(box);

    auto *grid = new QGridLayout;
    int r = 0;
    // Ranges mirror the clamps in EffectsSnapshot::apply_to_uniforms (web/state.rs).
    addFxRow(grid, r++, QStringLiteral("Pixelate"), QStringLiteral("pixelate"), 1.0, 32.0);
    addFxRow(grid, r++, QStringLiteral("RGB split"), QStringLiteral("rgb_split"), 0.0, 30.0);
    addFxRow(grid, r++, QStringLiteral("Hue shift"), QStringLiteral("hue_shift"), -180.0, 180.0);
    addFxRow(grid, r++, QStringLiteral("Saturation"), QStringLiteral("saturation"), -1.0, 1.0);
    addFxRow(grid, r++, QStringLiteral("Contrast"), QStringLiteral("contrast"), -1.0, 1.0);
    addFxRow(grid, r++, QStringLiteral("Vignette"), QStringLiteral("vignette"), 0.0, 1.5);
    v->addLayout(grid);

    auto *row = new QHBoxLayout;
    m_invert = new QCheckBox(QStringLiteral("Invert"));
    connect(m_invert, &QCheckBox::toggled, this,
            [this](bool on) { m_engine->setParam(QStringLiteral("invert"), on); });
    auto *reset = new QPushButton(QStringLiteral("Reset FX"));
    connect(reset, &QPushButton::clicked, this, [this] { m_engine->resetFx(); });
    row->addWidget(m_invert);
    row->addStretch();
    row->addWidget(reset);
    v->addLayout(row);
    return box;
}

void ControlPanel::addFxRow(QGridLayout *grid, int row, const QString &label,
                            const QString &param, double mn, double mx) {
    auto *name = new QLabel(label);
    auto *slider = new QSlider(Qt::Horizontal);
    slider->setRange(0, kSliderSteps);
    auto *val = new QLabel;
    val->setMinimumWidth(56);
    val->setAlignment(Qt::AlignRight | Qt::AlignVCenter);

    // User drags only: programmatic setValue() in onSnapshot() is wrapped in a
    // QSignalBlocker, so this never echoes the engine's own value back at it.
    connect(slider, &QSlider::valueChanged, this, [this, param, val, mn, mx](int s) {
        const double v = sliderToFloat(s, mn, mx);
        val->setText(fmt(v));
        m_engine->setParam(param, v);
    });

    grid->addWidget(name, row, 0);
    grid->addWidget(slider, row, 1);
    grid->addWidget(val, row, 2);
    m_fx.push_back(FxSlider{param, slider, val, mn, mx});
}

QWidget *ControlPanel::buildMasterAudio() {
    auto *box = new QGroupBox(QStringLiteral("Master audio"));
    auto *grid = new QGridLayout(box);

    grid->addWidget(new QLabel(QStringLiteral("Volume (dB)")), 0, 0);
    m_masterVol = new QSlider(Qt::Horizontal);
    m_masterVol->setRange(0, kSliderSteps);
    m_masterVolVal = new QLabel;
    m_masterVolVal->setMinimumWidth(56);
    m_masterVolVal->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    connect(m_masterVol, &QSlider::valueChanged, this, [this](int s) {
        const double v = sliderToFloat(s, -60.0, 6.0);
        m_masterVolVal->setText(fmt(v));
        m_engine->setMasterAudioParam(QStringLiteral("master_volume"), v);
    });
    grid->addWidget(m_masterVol, 0, 1);
    grid->addWidget(m_masterVolVal, 0, 2);

    m_limiter = new QCheckBox(QStringLiteral("Limiter"));
    connect(m_limiter, &QCheckBox::toggled, this,
            [this](bool on) { m_engine->setMasterAudioParam(QStringLiteral("limiter"), on); });
    grid->addWidget(m_limiter, 1, 0);

    grid->addWidget(new QLabel(QStringLiteral("Output")), 2, 0);
    m_meter = new QProgressBar;
    m_meter->setRange(0, kSliderSteps);
    m_meter->setTextVisible(false);
    grid->addWidget(m_meter, 2, 1, 1, 2);
    return box;
}

QWidget *ControlPanel::buildLayers() {
    auto *box = new QGroupBox(QStringLiteral("Layers"));
    auto *outer = new QVBoxLayout(box);
    auto *scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    auto *inner = new QWidget;
    m_layerLayout = new QVBoxLayout(inner);
    m_layerLayout->setAlignment(Qt::AlignTop);
    scroll->setWidget(inner);
    outer->addWidget(scroll);
    return box;
}

void ControlPanel::rebuildLayers(int count) {
    for (const LayerRow &row : m_layerRows)
        row.container->deleteLater();
    m_layerRows.clear();

    for (int i = 0; i < count; ++i) {
        LayerRow row;
        row.container = new QWidget;
        auto *v = new QVBoxLayout(row.container);
        v->setContentsMargins(4, 4, 4, 4);

        row.name = new QLabel;
        row.name->setStyleSheet(QStringLiteral("font-weight: bold;"));
        v->addWidget(row.name);

        auto *opRow = new QHBoxLayout;
        opRow->addWidget(new QLabel(QStringLiteral("Opacity")));
        row.opacity = new QSlider(Qt::Horizontal);
        row.opacity->setRange(0, kSliderSteps);
        row.opacityVal = new QLabel;
        row.opacityVal->setMinimumWidth(56);
        row.opacityVal->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        connect(row.opacity, &QSlider::valueChanged, this,
                [this, i, lbl = row.opacityVal](int s) {
                    const double v = sliderToFloat(s, 0.0, 1.0);
                    lbl->setText(fmt(v));
                    m_engine->setLayerParam(i, QStringLiteral("opacity"), v);
                });
        opRow->addWidget(row.opacity);
        opRow->addWidget(row.opacityVal);
        v->addLayout(opRow);

        auto *btnRow = new QHBoxLayout;
        row.visible = new QPushButton;
        row.visible->setCheckable(true);
        connect(row.visible, &QPushButton::clicked, this, [this, i] { m_engine->toggleVisibility(i); });
        row.pause = new QPushButton;
        row.pause->setCheckable(true);
        connect(row.pause, &QPushButton::clicked, this, [this, i] { m_engine->toggleLayerPause(i); });
        btnRow->addWidget(row.visible);
        btnRow->addWidget(row.pause);
        btnRow->addStretch();
        v->addLayout(btnRow);

        row.progress = new QProgressBar;
        row.progress->setRange(0, kSliderSteps);
        row.progress->setTextVisible(false);
        row.progress->setFixedHeight(6);
        v->addWidget(row.progress);

        row.meter = new QProgressBar;
        row.meter->setRange(0, kSliderSteps);
        row.meter->setTextVisible(false);
        row.meter->setFixedHeight(6);
        v->addWidget(row.meter);

        auto *line = new QFrame;
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);
        v->addWidget(line);

        m_layerLayout->addWidget(row.container);
        m_layerRows.push_back(row);
    }
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
    // --- master transport ---
    const bool paused = snap.value(QStringLiteral("paused")).toBool();
    m_pauseBtn->setText(paused ? QStringLiteral("Resume") : QStringLiteral("Pause"));

    // --- master FX (editing guard: never overwrite a slider being dragged) ---
    const QJsonObject fx = snap.value(QStringLiteral("effects")).toObject();
    for (const FxSlider &f : m_fx) {
        if (f.slider->isSliderDown())
            continue;
        const double v = fx.value(f.param).toDouble();
        QSignalBlocker block(f.slider);
        f.slider->setValue(floatToSlider(v, f.min, f.max));
        f.value->setText(fmt(v));
    }
    if (!m_invert->hasFocus()) {
        QSignalBlocker block(m_invert);
        m_invert->setChecked(fx.value(QStringLiteral("invert")).toBool());
    }

    // --- master audio ---
    if (!m_masterVol->isSliderDown()) {
        const double vol = snap.value(QStringLiteral("master_volume")).toDouble();
        QSignalBlocker block(m_masterVol);
        m_masterVol->setValue(floatToSlider(vol, -60.0, 6.0));
        m_masterVolVal->setText(fmt(vol));
    }
    if (!m_limiter->hasFocus()) {
        QSignalBlocker block(m_limiter);
        m_limiter->setChecked(snap.value(QStringLiteral("master_limiter")).toBool());
    }
    m_meter->setValue(static_cast<int>(snap.value(QStringLiteral("meter")).toDouble() * kSliderSteps));

    // --- layers (rebuild rows only when the count changes) ---
    const QJsonArray layers = snap.value(QStringLiteral("layers")).toArray();
    if (layers.size() != m_layerRows.size())
        rebuildLayers(static_cast<int>(layers.size()));
    for (int i = 0; i < layers.size() && i < m_layerRows.size(); ++i) {
        const QJsonObject l = layers.at(i).toObject();
        const LayerRow &row = m_layerRows.at(i);
        row.name->setText(l.value(QStringLiteral("filename")).toString());

        if (!row.opacity->isSliderDown()) {
            const double op = l.value(QStringLiteral("opacity")).toDouble();
            QSignalBlocker block(row.opacity);
            row.opacity->setValue(floatToSlider(op, 0.0, 1.0));
            row.opacityVal->setText(fmt(op));
        }

        // visible/pause buttons connect to clicked(), not toggled(), so
        // setChecked() here won't echo back as an action.
        const bool vis = l.value(QStringLiteral("visible")).toBool();
        row.visible->setText(vis ? QStringLiteral("Visible") : QStringLiteral("Hidden"));
        row.visible->setChecked(vis);

        const bool lpaused = l.value(QStringLiteral("paused")).toBool();
        row.pause->setText(lpaused ? QStringLiteral("Paused") : QStringLiteral("Playing"));
        row.pause->setChecked(!lpaused);

        row.progress->setValue(static_cast<int>(l.value(QStringLiteral("progress")).toDouble() * kSliderSteps));
        row.meter->setValue(static_cast<int>(l.value(QStringLiteral("meter")).toDouble() * kSliderSteps));
    }
}
