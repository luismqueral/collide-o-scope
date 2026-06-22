#include "MatrixCell.h"

#include "EngineClient.h"

#include <QColorDialog>
#include <QFontMetrics>
#include <QInputDialog>
#include <QJsonArray>
#include <QMouseEvent>
#include <QPainter>
#include <QSet>

#include <algorithm>
#include <cmath>

namespace {
// Palette mirrored from static/style.css :root.
const QColor kSurface(0x16, 0x16, 0x1c);
const QColor kSurface2(0x1c, 0x1c, 0x24);
const QColor kBorder(0x2a, 0x2a, 0x35);
const QColor kText(0xd8, 0xd8, 0xdc);
const QColor kTextDim(0x6a, 0x6a, 0x78);
const QColor kAccent(0x5a, 0x8f, 0xe6);
const QColor kAccentHover(0x7a, 0xa8, 0xf0);
const QColor kBg(0x0f, 0x0f, 0x12);

// Params that still apply to an audio-only column (no video). Everything else
// is blanked. Mirrors AUDIO_LAYER_PARAMS in static/matrix.js.
const QSet<QString> &audioLayerParams() {
    static const QSet<QString> s = {
        QStringLiteral("clip"),    QStringLiteral("speed"),     QStringLiteral("paused"),
        QStringLiteral("mute"),    QStringLiteral("volume"),    QStringLiteral("pan"),
        QStringLiteral("eq_low"),  QStringLiteral("eq_mid"),    QStringLiteral("eq_high"),
        QStringLiteral("delay_time"), QStringLiteral("delay_feedback"), QStringLiteral("delay_mix"),
    };
    return s;
}

int decimalsForStep(double step) {
    if (step >= 1.0)
        return 0;
    if (step >= 0.1)
        return 1;
    if (step >= 0.01)
        return 2;
    if (step >= 0.001)
        return 3;
    return 4;
}

QString fmtNum(double v, double step) { return QString::number(v, 'f', decimalsForStep(step)); }
} // namespace

MatrixCell::MatrixCell(const ParamDef *def, ColumnDesc col, EngineClient *engine,
                       const QStringList *library, QWidget *parent)
    : QWidget(parent), m_def(def), m_col(col), m_engine(engine), m_library(library) {
    setMinimumHeight(22);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    // Seed local value from the schema default so the first paint (before any
    // snapshot) shows sensible rest values.
    m_value = def->def;
    if (def->ptype == PType::Color)
        m_color = def->defColor;
    // Cursor reflects what a click/drag would do.
    if (isNa())
        setCursor(Qt::ArrowCursor);
    else if (def->ptype == PType::Float || def->ptype == PType::Bipolar)
        setCursor(Qt::SizeHorCursor);
    else
        setCursor(Qt::PointingHandCursor);
}

QSize MatrixCell::sizeHint() const { return QSize(64, 22); }

bool MatrixCell::applies() const { return channelApplies(*m_def, m_col.kind); }

bool MatrixCell::isNa() const {
    if (!applies())
        return true;
    // Video params on an audio-only layer column are blanked.
    if (m_col.kind == ColumnKind::Layer && m_audioOnly && !audioLayerParams().contains(m_def->key))
        return true;
    return false;
}

bool MatrixCell::isChanged() const {
    switch (m_def->ptype) {
    case PType::Float:
    case PType::Bipolar:
        return std::fabs(m_value - m_def->def) > 1e-6;
    case PType::Bool:
        return (m_value != 0.0) != (m_def->def != 0.0);
    case PType::Enum:
        return m_enumIndex != 0; // default is always options[0]
    case PType::Color:
        return m_color.compare(m_def->defColor, Qt::CaseInsensitive) != 0;
    case PType::Clip:
        return !m_clip.isEmpty();
    }
    return false;
}

// --- painting ---------------------------------------------------------------

void MatrixCell::paintEvent(QPaintEvent *) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);
    const QRectF full = rect();

    // Cell background + grid lines (left + bottom, like .mx-cell).
    p.fillRect(full, kSurface);
    p.setPen(kBorder);
    p.drawLine(full.topLeft(), full.bottomLeft());
    p.drawLine(full.bottomLeft(), full.bottomRight());

    if (isNa()) {
        p.setPen(kBorder.lighter(140));
        p.drawText(full, Qt::AlignCenter, QStringLiteral("\u2014")); // —
        return;
    }

    const bool changed = isChanged();
    const QRectF inner = full.adjusted(4, 3, -4, -3);

    switch (m_def->ptype) {
    case PType::Float:
    case PType::Bipolar: {
        // Scrub bar: fill proportional to value, numeric readout centred on top.
        QColor fill = kAccent;
        fill.setAlpha(changed ? 140 : 76);
        const double range = m_def->max - m_def->min;
        if (range > 0) {
            if (m_def->ptype == PType::Float) {
                const double t = std::clamp((m_value - m_def->min) / range, 0.0, 1.0);
                QRectF f = inner;
                f.setWidth(inner.width() * t);
                p.fillRect(f, fill);
            } else {
                // Bipolar: fill from centre outward toward the value.
                const double half = inner.width() / 2.0;
                const double cx = inner.left() + half;
                const double t = std::clamp(m_value / m_def->max, -1.0, 1.0);
                if (t >= 0)
                    p.fillRect(QRectF(cx, inner.top(), half * t, inner.height()), fill);
                else
                    p.fillRect(QRectF(cx + half * t, inner.top(), -half * t, inner.height()), fill);
                // Centre tick.
                p.setPen(kBorder.lighter(160));
                p.drawLine(QPointF(cx, inner.top()), QPointF(cx, inner.bottom()));
            }
        }
        p.setPen(changed ? kText : kTextDim);
        p.drawText(full, Qt::AlignCenter, fmtNum(m_value, m_def->step));
        break;
    }
    case PType::Bool: {
        const bool on = m_value != 0.0;
        const QString txt = on ? QStringLiteral("ON") : QStringLiteral("OFF");
        QRectF pill(0, 0, 34, 14);
        pill.moveCenter(full.center());
        QColor bg = on ? (changed ? kAccentHover : kAccent) : kSurface2;
        p.setPen(on ? Qt::NoPen : QPen(kBorder));
        p.setBrush(bg);
        p.drawRoundedRect(pill, 3, 3);
        p.setPen(on ? kBg : kTextDim);
        QFont f = p.font();
        f.setPointSize(8);
        f.setBold(true);
        p.setFont(f);
        p.drawText(pill, Qt::AlignCenter, txt);
        break;
    }
    case PType::Enum: {
        QString label;
        if (m_enumIndex >= 0 && m_enumIndex < m_def->options.size())
            label = m_def->options.at(m_enumIndex).label;
        p.setPen(changed ? kText : kTextDim);
        QFont f = p.font();
        f.setPointSize(9);
        p.setFont(f);
        p.drawText(full, Qt::AlignCenter, label);
        break;
    }
    case PType::Color: {
        QRectF sw(0, 0, std::min<double>(inner.width(), 28), inner.height());
        sw.moveCenter(full.center());
        QColor c(m_color.isEmpty() ? m_def->defColor : m_color);
        p.setPen(QPen(kBorder));
        p.setBrush(c.isValid() ? c : QColor(Qt::black));
        p.drawRoundedRect(sw, 2, 2);
        break;
    }
    case PType::Clip: {
        QFontMetrics fm(p.font());
        const QString elided = fm.elidedText(m_clip, Qt::ElideMiddle, int(inner.width()));
        p.setPen(m_clip.isEmpty() ? kTextDim : kText);
        p.drawText(full, Qt::AlignCenter, elided.isEmpty() ? QStringLiteral("\u2014") : elided);
        break;
    }
    }
}

// --- mouse ------------------------------------------------------------------

void MatrixCell::mousePressEvent(QMouseEvent *e) {
    if (isNa() || e->button() != Qt::LeftButton) {
        QWidget::mousePressEvent(e);
        return;
    }
    switch (m_def->ptype) {
    case PType::Float:
    case PType::Bipolar:
        m_scrubbing = true;
        m_pressX = e->pos().x();
        m_pressValue = m_value;
        break;
    case PType::Bool:
        toggleBool();
        break;
    case PType::Enum:
        cycleEnum();
        break;
    case PType::Color:
        pickColor();
        break;
    case PType::Clip:
        pickClip();
        break;
    }
}

void MatrixCell::mouseMoveEvent(QMouseEvent *e) {
    if (!m_scrubbing)
        return;
    const double range = m_def->max - m_def->min;
    const double w = std::max(1, width());
    const double delta = (double(e->pos().x() - m_pressX) / w) * range;
    double v = std::clamp(m_pressValue + delta, m_def->min, m_def->max);
    if (std::fabs(v - m_value) < 1e-9)
        return;
    m_value = v;
    update();
    sendNumeric(v);
}

void MatrixCell::mouseReleaseEvent(QMouseEvent *) { m_scrubbing = false; }

// --- click handlers ---------------------------------------------------------

void MatrixCell::toggleBool() {
    const bool now = m_value != 0.0;
    m_value = now ? 0.0 : 1.0; // optimistic; snapshot confirms
    update();
    sendBool(!now);
}

void MatrixCell::cycleEnum() {
    if (m_def->options.isEmpty())
        return;
    m_enumIndex = (m_enumIndex + 1) % m_def->options.size();
    update();
    sendEnum(m_def->options.at(m_enumIndex));
}

void MatrixCell::pickColor() {
    QColor init(m_color.isEmpty() ? m_def->defColor : m_color);
    const QColor c = QColorDialog::getColor(init.isValid() ? init : QColor(Qt::black), this,
                                            m_def->label);
    if (!c.isValid())
        return;
    m_color = c.name(); // #rrggbb
    update();
    sendColor(m_color);
}

void MatrixCell::pickClip() {
    if (!m_library || m_library->isEmpty())
        return;
    const int cur = std::max<int>(0, static_cast<int>(m_library->indexOf(m_clip)));
    bool ok = false;
    const QString choice = QInputDialog::getItem(this, QStringLiteral("Set clip"),
                                                 QStringLiteral("Clip:"), *m_library, cur, false,
                                                 &ok);
    if (!ok || choice.isEmpty())
        return;
    m_clip = choice;
    update();
    sendClip(choice);
}

// --- outbound routing -------------------------------------------------------

void MatrixCell::sendNumeric(double v) {
    if (m_col.kind == ColumnKind::Master) {
        switch (m_def->channels) {
        case Channels::MasterAudio:
            m_engine->setMasterAudioParam(m_def->key, v);
            break;
        case Channels::Ntsc:
            m_engine->setNtscParam(m_def->key, v);
            break;
        default: // Master / Both
            m_engine->setParam(m_def->key, v);
            break;
        }
    } else {
        m_engine->setLayerParam(m_col.layerIndex, m_def->key, v);
    }
}

void MatrixCell::sendBool(bool b) {
    if (m_col.kind == ColumnKind::Master) {
        switch (m_def->channels) {
        case Channels::MasterAudio:
            m_engine->setMasterAudioParam(m_def->key, b);
            break;
        case Channels::Ntsc:
            m_engine->setNtscParam(m_def->key, b);
            break;
        default:
            m_engine->setParam(m_def->key, b);
            break;
        }
        return;
    }
    // Layer column: visible/paused are dedicated toggle actions.
    if (m_def->key == QStringLiteral("visible"))
        m_engine->toggleVisibility(m_col.layerIndex);
    else if (m_def->key == QStringLiteral("paused"))
        m_engine->toggleLayerPause(m_col.layerIndex);
    else
        m_engine->setLayerParam(m_col.layerIndex, m_def->key, b);
}

void MatrixCell::sendEnum(const EnumOpt &o) {
    if (o.value.typeId() == QMetaType::QString) {
        // String enum (blend_mode) — layer only.
        m_engine->setLayerParam(m_col.layerIndex, m_def->key, o.value.toString());
        return;
    }
    const double v = o.value.toInt();
    if (m_col.kind == ColumnKind::Master) {
        if (m_def->channels == Channels::Ntsc)
            m_engine->setNtscParam(m_def->key, v);
        else
            m_engine->setParam(m_def->key, v); // e.g. grain_algo
    } else {
        m_engine->setLayerParam(m_col.layerIndex, m_def->key, v); // wave_axis/slice_axis/fit_mode
    }
}

void MatrixCell::sendColor(const QString &hex) {
    // Only chroma_color / chroma_bg_color exist — both layer params.
    m_engine->setLayerParam(m_col.layerIndex, m_def->key, hex);
}

void MatrixCell::sendClip(const QString &filename) {
    m_engine->setLayerClip(m_col.layerIndex, filename);
}

// --- inbound read-back ------------------------------------------------------

void MatrixCell::applyValue(const QJsonObject &snap) {
    // Resolve the owning layer object (and its audio-only flag) up front.
    QJsonObject layerObj;
    if (m_col.kind == ColumnKind::Layer) {
        const QJsonArray layers = snap.value(QStringLiteral("layers")).toArray();
        if (m_col.layerIndex >= layers.size())
            return; // stale: ControlPanel rebuilds on count change
        layerObj = layers.at(m_col.layerIndex).toObject();
        const bool ao = layerObj.value(QStringLiteral("audio_only")).toBool();
        if (ao != m_audioOnly) {
            m_audioOnly = ao;
            setCursor(isNa() ? Qt::ArrowCursor
                             : (m_def->ptype == PType::Float || m_def->ptype == PType::Bipolar)
                                   ? Qt::SizeHorCursor
                                   : Qt::PointingHandCursor);
        }
    }

    if (m_scrubbing || isNa()) {
        update();
        return;
    }

    // Locate the raw JSON value for this cell.
    const QString snapKey = m_def->snap.isEmpty() ? m_def->key : m_def->snap;
    QJsonValue raw;
    if (m_col.kind == ColumnKind::Layer) {
        raw = (m_def->ptype == PType::Clip) ? layerObj.value(QStringLiteral("filename"))
                                            : layerObj.value(m_def->key);
    } else {
        switch (m_def->channels) {
        case Channels::MasterAudio:
            raw = snap.value(snapKey); // top-level master_volume / master_limiter
            break;
        case Channels::Ntsc:
            raw = snap.value(QStringLiteral("ntsc")).toObject().value(m_def->key);
            break;
        default: // Master / Both
            raw = snap.value(QStringLiteral("effects")).toObject().value(snapKey);
            break;
        }
    }

    switch (m_def->ptype) {
    case PType::Float:
    case PType::Bipolar:
        m_value = raw.toDouble();
        break;
    case PType::Bool:
        m_value = raw.toBool() ? 1.0 : 0.0;
        break;
    case PType::Enum: {
        int idx = m_enumIndex;
        if (raw.isString()) {
            const QString s = raw.toString();
            for (int i = 0; i < m_def->options.size(); ++i)
                if (m_def->options.at(i).value.toString() == s) {
                    idx = i;
                    break;
                }
        } else {
            const int iv = int(std::lround(raw.toDouble()));
            for (int i = 0; i < m_def->options.size(); ++i)
                if (m_def->options.at(i).value.toInt() == iv) {
                    idx = i;
                    break;
                }
        }
        m_enumIndex = idx;
        break;
    }
    case PType::Color:
        m_color = raw.toString();
        break;
    case PType::Clip:
        m_clip = raw.toString();
        break;
    }
    update();
}
