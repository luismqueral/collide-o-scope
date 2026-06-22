#include "Schema.h"

#include <QHash>

// Transcribed verbatim from static/matrix-schema.js (MATRIX_GROUPS lines 17–190,
// LAYER_GROUPS lines 215–228, CHANNEL_APPLIES lines 197–203). Small builder
// helpers below keep each row readable while staying field-for-field with the JS.

namespace {

// Numeric rows (Float / Bipolar). `noRand` mirrors the JS `noRandom` flag.
ParamDef num(PType t, const char *key, const char *label, double mn, double mx,
             double step, double def, bool automatable, Channels ch, bool noRand = false) {
    ParamDef p;
    p.key = QString::fromLatin1(key);
    p.label = QString::fromLatin1(label);
    p.ptype = t;
    p.min = mn;
    p.max = mx;
    p.step = step;
    p.def = def;
    p.automatable = automatable;
    p.noRandom = noRand;
    p.channels = ch;
    return p;
}

// Enum rows. The default is always options[0] in this schema, so no separate
// def is needed; we just carry the option list + value type via QVariant.
ParamDef en(const char *key, const char *label, QVector<EnumOpt> opts, Channels ch) {
    ParamDef p;
    p.key = QString::fromLatin1(key);
    p.label = QString::fromLatin1(label);
    p.ptype = PType::Enum;
    p.options = std::move(opts);
    p.channels = ch;
    return p;
}

// Bool rows. `def` true→1.0 / false→0.0; `snap` overrides the snapshot key.
ParamDef boolean(const char *key, const char *label, bool def, Channels ch,
                 const char *snap = nullptr) {
    ParamDef p;
    p.key = QString::fromLatin1(key);
    p.label = QString::fromLatin1(label);
    p.ptype = PType::Bool;
    p.def = def ? 1.0 : 0.0;
    p.channels = ch;
    if (snap)
        p.snap = QString::fromLatin1(snap);
    return p;
}

ParamDef color(const char *key, const char *label, const char *def, Channels ch) {
    ParamDef p;
    p.key = QString::fromLatin1(key);
    p.label = QString::fromLatin1(label);
    p.ptype = PType::Color;
    p.defColor = QString::fromLatin1(def);
    p.channels = ch;
    return p;
}

ParamDef clip(const char *key, const char *label, Channels ch) {
    ParamDef p;
    p.key = QString::fromLatin1(key);
    p.label = QString::fromLatin1(label);
    p.ptype = PType::Clip;
    p.channels = ch;
    return p;
}

// Enum option shorthands: int-valued and string-valued.
EnumOpt opt(int value, const char *label) {
    return EnumOpt{QString::fromLatin1(label), QVariant(value)};
}
EnumOpt opt(const char *value, const char *label) {
    return EnumOpt{QString::fromLatin1(label), QVariant(QString::fromLatin1(value))};
}

QVector<Group> buildMatrixGroups() {
    const Channels Master = Channels::Master;
    const Channels Ntsc = Channels::Ntsc;
    const Channels Layer = Channels::Layer;
    const Channels Both = Channels::Both;
    const Channels MasterAudio = Channels::MasterAudio;
    const PType F = PType::Float;
    const PType B = PType::Bipolar;

    return {
        {QStringLiteral("LAYER"),
         {
             clip("clip", "clip", Layer),
             num(F, "opacity", "opacity", 0, 1, 0.01, 1, true, Layer),
             num(F, "speed", "speed", 0.25, 4, 0.25, 1, true, Layer),
             num(F, "fps", "fps", 1, 30, 1, 30, true, Layer),
             num(F, "loop_start", "loop in", 0, 1, 0.01, 0, false, Layer),
             num(F, "loop_end", "loop out", 0, 1, 0.01, 1, false, Layer),
             en("blend_mode", "blend",
                {opt("normal", "normal"), opt("screen", "screen"), opt("multiply", "multiply"),
                 opt("difference", "difference")},
                Layer),
             boolean("visible", "visible", true, Layer),
             boolean("paused", "paused", false, Layer),
         }},
        {QStringLiteral("AUDIO"),
         {
             num(F, "master_volume", "volume", -60, 6, 1, 0, false, MasterAudio, true),
             boolean("limiter", "limiter", true, MasterAudio, "master_limiter"),
             boolean("mute", "mute", false, Layer),
             num(F, "volume", "volume", -60, 6, 1, 0, false, Layer, true),
             num(B, "pan", "pan", -1, 1, 0.05, 0, false, Layer, true),
         }},
        {QStringLiteral("COLOR"),
         {
             num(B, "hue_shift", "hue", -180, 180, 1, 0, true, Both),
             num(B, "saturation", "sat", -1, 1, 0.01, 0, true, Both),
             num(B, "brightness", "bright", -1, 1, 0.01, 0, true, Both),
             num(B, "contrast", "contrast", -1, 1, 0.01, 0, true, Both),
             num(F, "rgb_split", "rgb split", 0, 30, 0.5, 0, true, Both),
             num(F, "posterize", "posterize", 0, 16, 1, 0, true, Both),
             boolean("invert", "invert", false, Both),
             num(F, "pixelate", "pixelate", 1, 32, 1, 1, true, Both),
         }},
        {QStringLiteral("MOTION"),
         {
             num(F, "breathe_scale", "breathe sc", 0, 0.05, 0.001, 0, true, Master),
             num(F, "breathe_rotation", "breathe rot", 0, 2, 0.05, 0, true, Master),
             num(F, "breathe_position", "breathe pos", 0, 0.02, 0.001, 0, true, Master),
             num(F, "grain_intensity", "grain", 0, 0.3, 0.005, 0, true, Master),
             num(F, "grain_size", "grain sz", 1, 4, 0.25, 1, true, Master),
             en("grain_algo", "grain algo",
                {opt(0, "Gaussian"), opt(1, "Perlin"), opt(2, "S&P"), opt(3, "Blue")}, Master),
             boolean("color_grain", "color grain", false, Master),
             num(F, "vignette", "vignette", 0, 1.5, 0.01, 0, true, Master),
             num(F, "color_drift", "color drift", 0, 0.02, 0.001, 0, true, Master),
         }},
        {QStringLiteral("WARP"),
         {
             num(F, "wave_amp", "wave amp", 0, 0.1, 0.001, 0, true, Layer),
             num(F, "wave_freq", "wave freq", 0, 50, 1, 0, true, Layer),
             num(F, "wave_speed", "wave spd", 0, 10, 0.1, 0, true, Layer),
             en("wave_axis", "wave axis", {opt(0, "H"), opt(1, "V"), opt(2, "Both")}, Layer),
             num(B, "swirl_angle", "swirl ang", -720, 720, 5, 0, true, Layer),
             num(F, "swirl_radius", "swirl rad", 0, 1, 0.01, 0, true, Layer),
             num(B, "bulge_strength", "bulge str", -1, 1, 0.01, 0, true, Layer),
             num(F, "bulge_radius", "bulge rad", 0.05, 1, 0.01, 0.05, true, Layer),
         }},
        {QStringLiteral("KEY"),
         {
             boolean("chroma_enable", "key on", false, Layer),
             color("chroma_color", "key color", "#00ff00", Layer),
             num(F, "chroma_threshold", "threshold", 0, 1, 0.01, 0.4, true, Layer),
             num(F, "chroma_smoothness", "smoothness", 0, 1, 0.01, 0.1, true, Layer),
             num(F, "chroma_spill", "spill", 0, 1, 0.01, 0, true, Layer),
             boolean("chroma_bg_enable", "bg on", false, Layer),
             color("chroma_bg_color", "bg color", "#000000", Layer),
         }},
        {QStringLiteral("SHIFT"),
         {
             num(F, "slice_intensity", "slice int", 0, 1, 0.01, 0, true, Layer),
             num(F, "slice_height", "slice ht", 1, 128, 1, 1, true, Layer),
             num(F, "slice_prob", "slice prob", 0, 1, 0.01, 0, true, Layer),
             num(F, "slice_speed", "slice spd", 0, 30, 1, 0, true, Layer),
             en("slice_axis", "slice axis", {opt(0, "H"), opt(1, "V"), opt(2, "Both")}, Layer),
             num(F, "block_size", "block sz", 4, 128, 1, 4, true, Layer),
             num(F, "block_intensity", "block int", 0, 1, 0.01, 0, true, Layer),
             num(F, "block_prob", "block prob", 0, 1, 0.01, 0, true, Layer),
             num(F, "block_speed", "block spd", 0, 30, 1, 0, true, Layer),
             num(F, "shift_chroma", "shift chr", 0, 1, 0.01, 0, true, Layer),
             num(F, "jitter_amount", "jitter amt", 0, 1, 0.01, 0, true, Layer),
             num(F, "jitter_speed", "jitter spd", 0, 30, 1, 0, true, Layer),
             num(F, "datamosh", "datamosh", 0, 1, 0.01, 0, true, Layer),
         }},
        {QStringLiteral("FEEDBACK"),
         {
             num(F, "feedback_persistence", "persist", 0, 1, 0.01, 0, true, Layer),
             num(F, "feedback_zoom", "fb zoom", 0.8, 1.2, 0.005, 1, true, Layer),
             num(B, "feedback_rotate", "fb rotate", -30, 30, 0.5, 0, true, Layer),
             num(F, "feedback_luma_key", "fb luma", 0, 1, 0.01, 0, true, Layer),
             num(F, "feedback_chroma", "fb chroma", 0, 1, 0.01, 0, true, Layer),
             num(F, "feedback_additive", "fb add", 0, 1, 0.01, 0, true, Layer),
         }},
        {QStringLiteral("TRANSFORM"),
         {
             num(B, "layer_x", "pos x", -1, 1, 0.01, 0, true, Layer),
             num(B, "layer_y", "pos y", -1, 1, 0.01, 0, true, Layer),
             num(F, "layer_scale", "scale", 0.1, 4, 0.01, 1, true, Layer),
             en("fit_mode", "fit", {opt(0, "Stretch"), opt(1, "Fit"), opt(2, "Fill")}, Layer),
         }},
        {QStringLiteral("AUDIO FX"),
         {
             num(B, "eq_low", "eq low", -24, 12, 1, 0, false, Layer, true),
             num(B, "eq_mid", "eq mid", -24, 12, 1, 0, false, Layer, true),
             num(B, "eq_high", "eq high", -24, 12, 1, 0, false, Layer, true),
             num(F, "delay_time", "delay ms", 0, 1000, 10, 0, false, Layer, true),
             num(F, "delay_feedback", "delay fb", 0, 0.95, 0.05, 0, false, Layer, true),
             num(F, "delay_mix", "delay mix", 0, 1, 0.05, 0, false, Layer, true),
         }},
        {QStringLiteral("VHS/NTSC"),
         {
             boolean("enabled", "vhs on", false, Ntsc),
             en("tape_speed", "tape", {opt(0, "SP"), opt(1, "LP"), opt(2, "EP")}, Ntsc),
             num(F, "chroma_loss", "chroma loss", 0, 0.01, 0.0005, 0, false, Ntsc),
             boolean("edge_wave_enabled", "edge on", false, Ntsc),
             num(F, "edge_wave_intensity", "edge int", 0, 20, 0.5, 0, false, Ntsc),
             num(F, "edge_wave_speed", "edge spd", 0, 10, 0.1, 0.5, false, Ntsc),
             boolean("head_switching_enabled", "head on", false, Ntsc),
             num(F, "head_switching_height", "head ht", 0, 24, 1, 8, false, Ntsc),
             num(B, "head_switching_shift", "head shift", -100, 100, 1, 0, false, Ntsc),
             boolean("tracking_noise_enabled", "track on", false, Ntsc),
             num(F, "tracking_noise_height", "track ht", 0, 120, 1, 24, false, Ntsc),
             num(F, "tracking_noise_wave", "track wave", 0, 50, 0.5, 0, false, Ntsc),
             num(F, "tracking_noise_snow", "track snow", 0, 1, 0.01, 0, false, Ntsc),
             num(F, "snow_intensity", "snow", 0, 1, 0.01, 0, false, Ntsc),
             num(F, "composite_noise_intensity", "comp noise", 0, 0.5, 0.005, 0, false, Ntsc),
             num(F, "luma_noise_intensity", "luma noise", 0, 0.2, 0.005, 0, false, Ntsc),
             num(F, "chroma_noise_intensity", "chr noise", 0, 0.5, 0.005, 0, false, Ntsc),
             num(F, "luma_smear", "luma smear", 0, 1, 0.01, 0, false, Ntsc),
             num(B, "composite_sharpening", "sharpen", -1, 2, 0.05, 0, false, Ntsc),
         }},
    };
}

QVector<LayerGroup> buildLayerGroups() {
    return {
        {QStringLiteral("SOURCE"),
         {"clip", "speed", "fps", "loop_start", "loop_end", "paused"}},
        {QStringLiteral("AUDIO"), {"mute", "volume", "pan"}},
        {QStringLiteral("AUDIO FX"),
         {"eq_low", "eq_mid", "eq_high", "delay_time", "delay_feedback", "delay_mix"}},
        {QStringLiteral("BLEND"), {"opacity", "blend_mode", "visible"}},
        {QStringLiteral("TRANSFORM"), {"layer_x", "layer_y", "layer_scale", "fit_mode"}},
        {QStringLiteral("WARP"),
         {"wave_amp", "wave_freq", "wave_speed", "wave_axis", "swirl_angle", "swirl_radius",
          "bulge_strength", "bulge_radius", "pixelate"}},
        {QStringLiteral("COLOR"),
         {"hue_shift", "saturation", "brightness", "contrast", "invert", "posterize",
          "shift_chroma", "rgb_split"}},
        {QStringLiteral("COLOR KEY"),
         {"chroma_enable", "chroma_color", "chroma_threshold", "chroma_smoothness", "chroma_spill",
          "chroma_bg_enable", "chroma_bg_color"}},
        {QStringLiteral("SLICE"),
         {"slice_intensity", "slice_height", "slice_prob", "slice_speed", "slice_axis"}},
        {QStringLiteral("BLOCKS"), {"block_size", "block_intensity", "block_prob", "block_speed"}},
        {QStringLiteral("GLITCH"), {"jitter_amount", "jitter_speed", "datamosh"}},
        {QStringLiteral("FEEDBACK"),
         {"feedback_persistence", "feedback_zoom", "feedback_rotate", "feedback_luma_key",
          "feedback_chroma", "feedback_additive"}},
    };
}

} // namespace

const QVector<Group> &matrixGroups() {
    static const QVector<Group> groups = buildMatrixGroups();
    return groups;
}

const QVector<LayerGroup> &layerGroups() {
    static const QVector<LayerGroup> groups = buildLayerGroups();
    return groups;
}

const ParamDef *findParam(const QString &key) {
    // Flat key→def index built once from MATRIX_GROUPS (the superset). Pointers
    // are stable because matrixGroups() returns a function-local static.
    static const QHash<QString, const ParamDef *> index = [] {
        QHash<QString, const ParamDef *> m;
        for (const Group &g : matrixGroups())
            for (const ParamDef &p : g.params)
                m.insert(p.key, &p);
        return m;
    }();
    return index.value(key, nullptr);
}

bool channelApplies(const ParamDef &def, ColumnKind colKind) {
    switch (def.channels) {
    case Channels::Both:
        return colKind == ColumnKind::Master || colKind == ColumnKind::Layer;
    case Channels::Ntsc:
        return colKind == ColumnKind::Master;
    case Channels::MasterAudio:
        return colKind == ColumnKind::Master;
    case Channels::Master:
        return colKind == ColumnKind::Master;
    case Channels::Layer:
        return colKind == ColumnKind::Layer;
    }
    return false;
}
