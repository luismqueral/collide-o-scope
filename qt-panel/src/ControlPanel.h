#pragma once

#include <QJsonObject>
#include <QStringList>
#include <QUrl>
#include <QVector>
#include <QWidget>

class EngineClient;
class MatrixCell;
class QGridLayout;
class QLabel;
class QProgressBar;
class QPushButton;
class QScrollArea;

// The transposed-matrix control panel. Layout mirrors the browser:
//   * a centre layer grid — rows are parameters (grouped per LAYER_GROUPS),
//     columns are layers, each a stack of MatrixCells with a header (name / ×
//     remove / progress / meter) and a trailing "+" add-column button;
//   * a right Master panel — the MATRIX_GROUPS params that apply to the master
//     column (FX / audio / VHS), one column of MatrixCells, built once.
//
// Outbound edits happen inside each MatrixCell (it owns its EngineClient route).
// Inbound, onSnapshot() fans the ~30fps AppSnapshot out to every cell's
// applyValue() (which self-guards while scrubbing) and refreshes the live
// headers/progress/meters. The layer grid is rebuilt only when the layer count
// changes (the POC's rebuildLayers pattern).
class ControlPanel : public QWidget {
    Q_OBJECT
public:
    explicit ControlPanel(QUrl engineUrl, QWidget *parent = nullptr);

private slots:
    void onSnapshot(const QJsonObject &snap);
    void onConnectionChanged(bool connected);

private:
    // Shell.
    QWidget *buildConnectionBar();
    QWidget *buildTransport();

    // Matrix.
    QWidget *buildLayerPanel();  // scroll host for the layer grid
    QWidget *buildMasterPanel(); // the master column (built once)
    void rebuildLayerGrid(int count);
    QWidget *makeLayerHeader(int index);
    void addLayerViaPicker();

    EngineClient *m_engine = nullptr;

    // Connection bar.
    QLabel *m_statusDot = nullptr;
    QLabel *m_statusText = nullptr;

    // Master transport.
    QPushButton *m_pauseBtn = nullptr;

    // Layer grid (rebuilt when the layer count changes).
    QScrollArea *m_layerScroll = nullptr;
    QWidget *m_layerInner = nullptr;
    QGridLayout *m_layerGrid = nullptr;
    QPushButton *m_addColBtn = nullptr;
    int m_layerCount = -1;

    // Per-layer column header widgets, kept for live updates.
    struct LayerHeader {
        QWidget *container = nullptr;
        QLabel *title = nullptr;
        QLabel *filename = nullptr;
        QProgressBar *progress = nullptr;
        QProgressBar *meter = nullptr;
    };
    QVector<LayerHeader> m_layerHeaders;
    QVector<MatrixCell *> m_layerCells;

    // Master column.
    QVector<MatrixCell *> m_masterCells;
    QProgressBar *m_masterMeter = nullptr;

    // Latest library filename list (drives the add/swap clip pickers). Cells
    // hold a pointer to this member, so it must outlive them (it does).
    QStringList m_library;
};
