#pragma once

#include "Schema.h"

#include <QJsonObject>
#include <QString>
#include <QWidget>

class EngineClient;

// Identifies which matrix column a cell lives in. For layer columns,
// `layerIndex` is the index into the snapshot's `layers` array.
struct ColumnDesc {
    ColumnKind kind = ColumnKind::Layer;
    int layerIndex = 0;
};

// One custom-painted cell at the intersection of a parameter row and a column.
// It renders type-specifically (scrub bar / pill / enum label / swatch /
// filename), routes edits to the right EngineClient action by channel×column,
// and pulls live values back from each snapshot via applyValue().
//
// The editing guard: while the user is scrubbing this cell (m_scrubbing), or
// the layer index no longer exists, applyValue() is a no-op so incoming 30fps
// snapshots never yank the value out from under a drag. Programmatic applyValue
// only sets the local value + repaints — it never calls the engine, so there is
// no echo loop (no QSignalBlocker dance needed).
class MatrixCell : public QWidget {
    Q_OBJECT
public:
    MatrixCell(const ParamDef *def, ColumnDesc col, EngineClient *engine,
               const QStringList *library, QWidget *parent = nullptr);

    // Push the latest snapshot value into this cell (skipped while scrubbing).
    void applyValue(const QJsonObject &snap);

    QSize sizeHint() const override;

protected:
    void paintEvent(QPaintEvent *) override;
    void mousePressEvent(QMouseEvent *) override;
    void mouseMoveEvent(QMouseEvent *) override;
    void mouseReleaseEvent(QMouseEvent *) override;

private:
    // True if this (param, column) combination is a real control rather than an
    // "—" N/A blank (channel mismatch, or a video param on an audio-only layer).
    bool applies() const;
    bool isNa() const;

    // Outbound routing by def.channels × col.kind.
    void sendNumeric(double v);
    void sendBool(bool b);
    void sendEnum(const EnumOpt &o);
    void sendColor(const QString &hex);
    void sendClip(const QString &filename);

    // Whether the cell currently differs from its schema default (drives the
    // dim-vs-bright "changed" styling).
    bool isChanged() const;

    // Click handlers per ptype.
    void cycleEnum();
    void toggleBool();
    void pickColor();
    void pickClip();

    const ParamDef *m_def = nullptr;
    ColumnDesc m_col;
    EngineClient *m_engine = nullptr;
    const QStringList *m_library = nullptr; // owned by ControlPanel; updated each snapshot

    // Local value state (only the field matching m_def->ptype is meaningful).
    double m_value = 0.0;   // float/bipolar value, or bool as 1/0
    int m_enumIndex = 0;    // index into m_def->options
    QString m_color;        // color hex
    QString m_clip;         // clip filename
    bool m_audioOnly = false; // owning layer is audio-only (blanks video rows)

    // Scrub state for float/bipolar.
    bool m_scrubbing = false;
    int m_pressX = 0;
    double m_pressValue = 0.0;
};
