#pragma once

#include <QString>
#include <QStringList>
#include <QVariant>
#include <QVector>

// C++ port of static/matrix-schema.js — the single source of truth for the
// transposed parameter matrix (rows = params, columns = channels). We mirror it
// here so ranges/defaults/options stay single-sourced exactly as the JS does:
// MATRIX_GROUPS is the full superset (and drives the Master column); LAYER_GROUPS
// is a decoupled re-grouping of the same param keys for the per-layer grid.
//
// Keep this transcription field-for-field with matrix-schema.js. The keys must
// match EffectsSnapshot / NtscSnapshot / LayerSnapshot in src/web/state.rs.

// The control widget a param renders as. Mirrors the JS `ptype` strings.
enum class PType {
    Float,   // 0..max scrub bar (fill left→value)
    Bipolar, // -max..max scrub bar (fill from centre)
    Enum,    // click-to-cycle labelled options
    Bool,    // ON/OFF pill
    Color,   // swatch → QColorDialog
    Clip,    // filename → QInputDialog picker
};

// Which column kinds a param applies to. Mirrors the JS `channels` strings.
enum class Channels {
    Master,      // master FX bus only
    Ntsc,        // VHS/NTSC — shown under Master (global effect)
    Layer,       // per-layer only
    Both,        // master FX *and* per-layer
    MasterAudio, // master volume/limiter — shown under Master
};

// The kind of a single matrix column. Master is the global bus; Layer columns
// each carry the index into the snapshot's `layers` array.
enum class ColumnKind {
    Master,
    Layer,
};

// One enum option. `value` is an int (int-enums like grain_algo / wave_axis) or
// a QString (blend_mode), matching the JS `{value, label}` shape.
struct EnumOpt {
    QString label;
    QVariant value;
};

// One parameter row. Field names/semantics mirror the JS ParamDef.
struct ParamDef {
    QString key;   // snapshot/action key
    QString label; // short row label
    PType ptype = PType::Float;
    double min = 0.0;
    double max = 1.0;
    double step = 0.01;
    double def = 0.0;        // numeric/bool default (bool: 1=true, 0=false). Enums use options[0].
    QString defColor;        // color only: default hex string (e.g. "#00ff00")
    QVector<EnumOpt> options; // enum only
    bool automatable = false;
    bool noRandom = false;
    Channels channels = Channels::Layer;
    QString snap; // snapshot-key override where it differs from key (e.g. limiter→master_limiter)
};

// A MATRIX_GROUPS entry: a named group of param rows (drives the Master column).
struct Group {
    QString name;
    QVector<ParamDef> params;
};

// A LAYER_GROUPS entry: a named group of param *keys* (drives the layer grid).
// Keys resolve back to the shared ParamDefs in MATRIX_GROUPS via findParam().
struct LayerGroup {
    QString name;
    QStringList keys;
};

// MATRIX_GROUPS — the full superset of params, in master-column order.
const QVector<Group> &matrixGroups();

// LAYER_GROUPS — the per-layer grid's own grouping/order (keys only).
const QVector<LayerGroup> &layerGroups();

// Flat key→def lookup built from MATRIX_GROUPS (the superset). Returns nullptr
// for an unknown key. The returned pointer is stable for the program's lifetime.
const ParamDef *findParam(const QString &key);

// Port of CHANNEL_APPLIES: does a param row apply to a given column kind?
bool channelApplies(const ParamDef &def, ColumnKind colKind);
