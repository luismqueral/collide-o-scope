#pragma once

#include <QJsonObject>
#include <QUrl>
#include <QVector>
#include <QWidget>

class EngineClient;
class QCheckBox;
class QGridLayout;
class QLabel;
class QProgressBar;
class QPushButton;
class QSlider;
class QVBoxLayout;

// One master-FX slider, paired with its numeric readout label and the float
// range it maps onto (the integer QSlider runs 0..kSliderSteps internally).
struct FxSlider {
    QString param;   // wire param name sent via set_param
    QSlider *slider = nullptr;
    QLabel *value = nullptr;
    double min = 0.0;
    double max = 1.0;
};

// The widgets making up one row of the layer strip. Kept so onSnapshot() can
// push live values into them each frame without rebuilding the row.
struct LayerRow {
    QWidget *container = nullptr;
    QLabel *name = nullptr;
    QSlider *opacity = nullptr;
    QLabel *opacityVal = nullptr;
    QPushButton *visible = nullptr;
    QPushButton *pause = nullptr;
    QProgressBar *progress = nullptr;
    QProgressBar *meter = nullptr;
};

// The whole control panel window. Builds its widgets once, then keeps them in
// sync with the engine: outbound, each control calls an EngineClient helper;
// inbound, onSnapshot() repaints displayed values from the ~30fps AppSnapshot.
class ControlPanel : public QWidget {
    Q_OBJECT
public:
    explicit ControlPanel(QUrl engineUrl, QWidget *parent = nullptr);

private slots:
    void onSnapshot(const QJsonObject &snap);
    void onConnectionChanged(bool connected);

private:
    QWidget *buildConnectionBar();
    QWidget *buildTransport();
    QWidget *buildMasterFx();
    QWidget *buildMasterAudio();
    QWidget *buildLayers();
    void addFxRow(QGridLayout *grid, int row, const QString &label,
                  const QString &param, double min, double max);
    void rebuildLayers(int count);

    EngineClient *m_engine = nullptr;

    // Connection bar
    QLabel *m_statusDot = nullptr;
    QLabel *m_statusText = nullptr;

    // Master transport
    QPushButton *m_pauseBtn = nullptr;

    // Master FX
    QVector<FxSlider> m_fx;
    QCheckBox *m_invert = nullptr;

    // Master audio
    QSlider *m_masterVol = nullptr;
    QLabel *m_masterVolVal = nullptr;
    QCheckBox *m_limiter = nullptr;
    QProgressBar *m_meter = nullptr;

    // Layer strip (rows rebuilt only when the layer count changes)
    QVBoxLayout *m_layerLayout = nullptr;
    QVector<LayerRow> m_layerRows;
};
