# Richard Cigarette Refactor Plan: Python → Node.js/TypeScript

> **Status**: Planning  
> **Created**: 2025-12-02  
> **Goal**: Modular, TypeScript-based video synthesis with optional Next.js UI

---

## Overview

Refactoring the Python video blender scripts into a modular TypeScript architecture. The core insight: **treatments as plugins** that can be composed, configured, and swapped.

### Current State (Python)

| Script | Key Features |
|--------|-------------|
| `blend-video.py` | Basic colorkey overlay |
| `blend-video-alt.py` | Random trims, ML color extraction (K-means, rembg, luminance modes) |
| `blend-video-multi-vid.py` | Multi-layer compositing, audio mixing, looping |

All use FFmpeg under the hood—that won't change. Node.js will orchestrate FFmpeg with a modular architecture.

---

## Architecture Decision: Library + CLI + UI

**Library-first** with CLI as thin wrapper. This enables:

| Use Case | How It Works |
|----------|--------------|
| Quick render | `npx synth --preset classic` |
| CI/CD | Same CLI in GitHub Actions |
| UI | Import library in Next.js server actions |
| Programmatic | `import { createPipeline } from '@/lib/synth'` |

---

## Directory Structure

```
richard-cig-video-synth/
├── src/
│   ├── lib/
│   │   └── synth/
│   │       ├── index.ts                    # Named exports only
│   │       ├── types.ts                    # Types grouped at top
│   │       │
│   │       ├── core/
│   │       │   ├── pipeline.ts             # Fluent builder API
│   │       │   ├── video-source.ts         # Video file handling
│   │       │   ├── filter-chain-builder.ts # FFmpeg filter_complex
│   │       │   └── renderer.ts             # Executes FFmpeg
│   │       │
│   │       ├── treatments/                 # Modular treatments
│   │       │   ├── types.ts                # Treatment interface
│   │       │   ├── color-key.ts
│   │       │   ├── palette-key.ts
│   │       │   ├── kmeans-key.ts
│   │       │   ├── luminance-key.ts
│   │       │   └── index.ts                # Registry
│   │       │
│   │       ├── composers/
│   │       │   ├── types.ts
│   │       │   ├── overlay.ts
│   │       │   └── index.ts
│   │       │
│   │       ├── audio/
│   │       │   ├── mixer.ts
│   │       │   └── looper.ts
│   │       │
│   │       ├── postprocess/                # Post-blend pipeline
│   │       │   ├── analog.ts               # Grain, breathing, color drift
│   │       │   ├── color-grade.ts          # Color correction + presets
│   │       │   ├── audio-mixer.ts          # Layer audio with panning
│   │       │   └── stabilizer.ts           # Optical flow smoothing
│   │       │
│   │       ├── ai/                         # AI/ML integrations
│   │       │   ├── frame-processor.ts      # SD/fal.ai frame-by-frame
│   │       │   ├── keyframe-propagation.ts # Efficient SD with flow
│   │       │   └── providers/
│   │       │       ├── fal.ts
│   │       │       └── replicate.ts
│   │       │
│   │       └── utils/
│   │           ├── ffprobe.ts
│   │           ├── frame-extractor.ts
│   │           └── random.ts
│   │
│   ├── cli/
│   │   ├── index.ts                        # Entry point
│   │   └── commands/
│   │       ├── render.ts
│   │       └── upload.ts
│   │
│   └── app/                                # Next.js (phase 2)
│       ├── page.tsx
│       ├── components/
│       │   └── synth-controls/
│       └── api/
│           └── render/
│               └── route.ts
│
├── presets/                                # JSON configs
│   ├── classic.json
│   └── experimental.json
│
├── input/
│   ├── video/
│   └── audio/
│
├── output/
│
├── docs/
│   ├── HOW-IT-WORKS.md
│   ├── ARCHITECTURE.md
│   └── REFACTOR-PLAN.md                   # This file
│
├── CHANGELOG.md
├── package.json
└── tsconfig.json
```

---

## Core Types

```typescript
// src/lib/synth/types.ts

// ============================================
// Core Pipeline Types
// ============================================

export interface VideoSource {
  path: string;
  duration: number;
  resolution: { width: number; height: number };
  hasAudio: boolean;
}

export interface TrimConfig {
  startTime: number;
  duration: number;
}

export interface RenderConfig {
  outputDir: string;
  size: [number, number];
  fps: number;
  codec: 'libx264' | 'libx265';
  crf: number;
}

// ============================================
// Treatment Types
// ============================================

export interface Treatment {
  /** Unique identifier for this treatment */
  name: string;
  
  /**
   * Generate FFmpeg filter string for a video stream.
   * Called once per video layer (except base layer).
   */
  apply: (
    streamRef: string,
    context: TreatmentContext
  ) => FilterFragment;
  
  /**
   * Optional async setup - e.g., extract frame for K-means analysis.
   * Called before apply() if defined.
   */
  prepare?: (videoPath: string) => Promise<void>;
}

export interface TreatmentContext {
  videoPath: string;
  layerIndex: number;
  duration: number;
  resolution: { width: number; height: number };
}

export interface FilterFragment {
  /** The filter string, e.g., "colorkey=0xFFFFFF:0.3:0.1" */
  filter: string;
  /** Optional debug info for logging */
  debug?: string;
}

// ============================================
// Preset Types
// ============================================

export interface Preset {
  name: string;
  videos: {
    count: number;
    hdOnly?: boolean;
    minResolution?: number;
  };
  duration: {
    min: number;
    max: number;
  } | number;
  treatments: TreatmentConfig[];
  composer: 'overlay' | 'blend' | 'grid';
  audio: {
    mixCount: number;
    normalize?: boolean;
  };
  output: Partial<RenderConfig>;
}

export type TreatmentConfig = 
  | { type: 'colorkey'; color: string; similarity?: number; blend?: number }
  | { type: 'palette'; colors: string[]; count?: number }
  | { type: 'kmeans'; colorsToExtract?: number; colorsToKey?: number }
  | { type: 'luminance'; target: 'lights' | 'darks' | 'auto'; threshold?: number };

// ============================================
// Post-Processing Types (Phase 3)
// ============================================

export interface AnalogConfig {
  grain?: {
    intensity: number;      // 0.05-0.20 typical
    size: 1 | 2 | 3;        // 1=fine, 2=coarse, 3=chunky
    algorithm: 'gaussian' | 'perlin' | 'salt_pepper' | 'blue';
    color?: boolean;        // chromatic grain
  };
  breathing?: {
    scale: number;          // 0.01-0.03 typical (±1-3% zoom)
    rotation?: number;      // degrees
    position?: number;      // drift amount
  };
  colorDrift?: number;      // RGB channel separation (0.005 typical)
  vignette?: number;        // corner darkening (0.2-0.4 typical)
}

export interface ColorGradeConfig {
  preset?: 'vintage' | 'vhs' | 'faded_film' | 'high_contrast' | 'cool_desat';
  // OR custom settings:
  temperature?: number;     // 3000=warm, 6500=neutral, 9000=cool
  saturation?: number;      // 0.0-2.0 (1.0 = unchanged)
  contrast?: number;        // 0.5-2.0 (1.0 = unchanged)
  brightness?: number;      // -0.5-0.5 (0 = unchanged)
  gamma?: number;           // 0.5-2.0 (1.0 = unchanged)
  colorBalance?: {
    shadows: { r: number; g: number; b: number };
    highlights: { r: number; g: number; b: number };
  };
}

export interface AudioMixConfig {
  tracks: Array<{
    source: string;         // path or 'random'
    pan: number;            // -1 (left) to 1 (right)
    volume: number;         // 0.0-2.0
    startOffset?: number;   // random offset if not specified
  }>;
  normalize?: boolean;
}
```

---

## Pipeline API (Fluent Builder)

```typescript
// src/lib/synth/core/pipeline.ts

import type { VideoSource, Treatment, RenderConfig, Preset } from '../types';

/**
 * Fluent builder for video synthesis pipelines.
 * 
 * Each method returns `this` for chaining. Call `render()` at the end
 * to execute the pipeline and generate output.
 * 
 * @example
 * ```ts
 * const result = await createPipeline()
 *   .selectVideos({ count: 3, hdOnly: true })
 *   .randomTrim({ durationRange: [60, 120] })
 *   .applyTreatment(colorKeyTreatment({ color: '0xFFFFFF' }))
 *   .compose('overlay')
 *   .render({ size: [800, 800], fps: 18 });
 * ```
 */
export const createPipeline = (options?: PipelineOptions) => {
  // Internal state
  let selectedVideos: VideoSource[] = [];
  let treatments: Treatment[] = [];
  let composerType: 'overlay' | 'blend' | 'grid' = 'overlay';
  let audioConfig = { mixCount: 2, normalize: true };
  let trimConfig = { min: 60, max: 120 };
  
  const pipeline = {
    /**
     * Select random videos from the input directory.
     * Optionally filter by resolution (HD mode).
     */
    selectVideos(config: {
      count: number;
      hdOnly?: boolean;
      minResolution?: number;
      directory?: string;
    }) {
      // Implementation: scan directory, filter, random sample
      return pipeline;
    },
    
    /**
     * Set the duration range for random trimming.
     * Each video gets a random start point within its length.
     */
    randomTrim(config: { durationRange: [number, number] }) {
      trimConfig = { min: config.durationRange[0], max: config.durationRange[1] };
      return pipeline;
    },
    
    /**
     * Add a treatment to apply to overlay layers.
     * Treatments are applied in order (can stack multiple).
     */
    applyTreatment(treatment: Treatment) {
      treatments.push(treatment);
      return pipeline;
    },
    
    /**
     * Set how videos are composited together.
     * - 'overlay': Stack with transparency (default)
     * - 'blend': FFmpeg blend modes
     * - 'grid': Side-by-side / tiled
     */
    compose(type: 'overlay' | 'blend' | 'grid') {
      composerType = type;
      return pipeline;
    },
    
    /**
     * Configure audio mixing from source videos.
     */
    mixAudio(config: { count: number; normalize?: boolean }) {
      audioConfig = { normalize: true, ...config };
      return pipeline;
    },
    
    /**
     * Execute the pipeline and render output.
     * Returns the path to the generated video.
     */
    async render(config: Partial<RenderConfig>): Promise<{ outputPath: string }> {
      // Build FFmpeg command and execute
      return { outputPath: '' };
    },
  };
  
  return pipeline;
};

/**
 * Create a pipeline from a preset JSON file.
 */
export const fromPreset = async (presetPath: string) => {
  // Load JSON, validate with Zod, build pipeline
};
```

---

## Treatment Examples

### Color Key Treatment

```typescript
// src/lib/synth/treatments/color-key.ts

import type { Treatment, FilterFragment, TreatmentContext } from '../types';

export interface ColorKeyOptions {
  /** Hex color to key out, e.g., '0xFFFFFF' for white */
  color: string;
  /** How similar a pixel must be to match (0.0-1.0) */
  similarity?: number;
  /** Edge softness (0.0-1.0) */
  blend?: number;
}

/**
 * Fixed color keying treatment.
 * Makes a specific color transparent, revealing layers below.
 * 
 * @example
 * ```ts
 * // Key out white pixels
 * pipeline.applyTreatment(colorKeyTreatment({ color: '0xFFFFFF' }))
 * 
 * // Key out black with softer edges
 * pipeline.applyTreatment(colorKeyTreatment({ 
 *   color: '0x000000', 
 *   similarity: 0.4,
 *   blend: 0.2 
 * }))
 * ```
 */
export const colorKeyTreatment = (options: ColorKeyOptions): Treatment => {
  const { color, similarity = 0.3, blend = 0.1 } = options;
  
  return {
    name: 'colorkey',
    
    apply: (streamRef: string, context: TreatmentContext): FilterFragment => {
      const filter = `colorkey=${color}:${similarity}:${blend}`;
      
      return {
        filter,
        debug: `Layer ${context.layerIndex}: keying ${color}`,
      };
    },
  };
};
```

### Palette Key Treatment

```typescript
// src/lib/synth/treatments/palette-key.ts

import type { Treatment } from '../types';
import { pickRandom } from '../utils/random';

export interface PaletteKeyOptions {
  /** Array of colors to randomly choose from */
  colors: string[];
  /** How many colors to key per video (default: 1) */
  count?: number;
  similarity?: number;
  blend?: number;
}

/**
 * Palette-based colorkeying.
 * Randomly selects colors from a curated palette for each video layer.
 */
export const paletteKeyTreatment = (options: PaletteKeyOptions): Treatment => {
  const { colors, count = 1, similarity = 0.3, blend = 0.1 } = options;
  
  return {
    name: 'palette-key',
    
    apply: (streamRef, context) => {
      const selectedColors = pickRandom(colors, count);
      
      // Chain multiple colorkey filters if count > 1
      const filter = selectedColors
        .map(color => `colorkey=${color}:${similarity}:${blend}`)
        .join(',');
      
      return {
        filter,
        debug: `Layer ${context.layerIndex}: palette ${selectedColors.join(', ')}`,
      };
    },
  };
};

// Pre-built palettes
export const PALETTES = {
  classic: ['0xFFFFFF', '0x000000'],
  neutrals: ['0xFFFFFF', '0x000000', '0x808080', '0xC0C0C0'],
  warm: ['0xFFFFFF', '0xFF6B35', '0xFFA500', '0xFFD700'],
  cool: ['0x000000', '0x0066CC', '0x00CED1', '0x228B22'],
} as const;
```

---

## Python Integration (Hybrid Approach)

For ML-heavy features, use **progressive enhancement**:

| Feature | Pure Node | Python Fallback |
|---------|-----------|-----------------|
| Fixed colorkey | ✅ | — |
| Palette selection | ✅ | — |
| Luminance analysis | ✅ (sharp) | — |
| K-means colors | ✅ (ml-kmeans) | Optional |
| Background removal | ⚠️ Limited | rembg (much better) |

```typescript
// Example: graceful fallback when Python not available

const isRembgAvailable = (): boolean => {
  try {
    execSync('python3 -c "import rembg"', { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
};

export const backgroundRemovalTreatment = (options: BGRemovalOptions): Treatment => {
  if (!isRembgAvailable()) {
    console.warn(
      '⚠️ rembg not installed. Using luminance fallback.\n' +
      '   Install with: pip install rembg'
    );
    return luminanceKeyTreatment({ target: 'auto' });
  }
  
  // Full rembg implementation...
};
```

---

## Next.js UI Vision (Phase 2)

### Server Action Integration

```typescript
// src/app/api/render/route.ts

import { createPipeline } from '@/lib/synth';
import { colorKeyTreatment } from '@/lib/synth/treatments';

export const POST = async (request: Request) => {
  const body = await request.json();
  const config = renderConfigSchema.parse(body);
  
  const result = await createPipeline()
    .selectVideos({ count: config.videoCount, hdOnly: config.hdOnly })
    .randomTrim({ durationRange: [config.minDuration, config.maxDuration] })
    .applyTreatment(colorKeyTreatment({ color: config.keyColor }))
    .compose('overlay')
    .render({ size: config.size, fps: config.fps });
  
  return Response.json({ outputPath: result.outputPath });
};
```

### UI Features (Future)

- **Video browser**: Thumbnails of `input/video/` with selection
- **Treatment picker**: Visual color pickers, sliders for similarity/blend
- **Preview**: Extract frames and show approximate result before full render
- **Queue**: Background rendering with progress (server-sent events)
- **Gallery**: Browse generated outputs

---

## Dependencies

| Purpose | Library |
|---------|---------|
| FFmpeg wrapper | `fluent-ffmpeg` |
| Image processing | `sharp` |
| K-means clustering | `ml-kmeans` |
| CLI | `commander` |
| Config validation | `zod` |
| YouTube upload | `@googleapis/youtube` |

---

## Implementation Phases

### Phase 1: Core Pipeline ✳️
- [ ] Project scaffolding (package.json, tsconfig)
- [ ] Core types and interfaces
- [ ] `Pipeline` builder
- [ ] `VideoSource` with ffprobe integration
- [ ] `FilterChainBuilder` for FFmpeg filter_complex
- [ ] `Renderer` to execute FFmpeg
- [ ] `colorKeyTreatment` (port blend-video.py)

### Phase 2: Treatment Library
- [ ] `paletteKeyTreatment` (random from presets)
- [ ] `luminanceKeyTreatment` (lights/darks)
- [ ] `kmeansKeyTreatment` (frame extraction + clustering)
- [ ] Treatment composition (stacking multiple)

### Phase 3: Post-Blend Pipeline (NEW)
- [ ] **Analog Processor** - Add organic character to output
  - [ ] Film grain (gaussian, perlin, salt_pepper, blue noise algorithms)
  - [ ] Grain intensity and size controls
  - [ ] Frame breathing (subtle random scale per frame)
  - [ ] Color drift (RGB channel separation)
  - [ ] Vignette effect
- [ ] **Color Correction** - FFmpeg-based color grading
  - [ ] Temperature (warm/cool)
  - [ ] Saturation, contrast, brightness, gamma
  - [ ] Color balance (shadows/highlights per RGB)
  - [ ] Built-in presets (vintage, VHS, faded_film, high_contrast, cool_desat)
  - [ ] RGB curves support
- [ ] **Audio Mixer** - Layer audio tracks onto video
  - [ ] Pick random audio from input library
  - [ ] Stereo panning (L/C/R positioning)
  - [ ] Random time offsets for variation
  - [ ] Volume normalization
  - [ ] Layer specific audio files with custom mix

### Phase 4: Advanced Features
- [ ] `backgroundRemovalTreatment` (rembg integration)
- [ ] Multiple composers (grid, blend modes)
- [ ] Generative frame processing (SD/fal.ai integration)
- [ ] Frame stabilization (optical flow, keyframe propagation)

### Phase 5: CLI + Presets
- [ ] `synth render` command
- [ ] `synth upload` command (YouTube)
- [ ] JSON preset loading
- [ ] Preset validation with Zod
- [ ] `synth sweep` command - Parameter sweep with HTML comparison output
  - Run multiple configurations (strength, guidance, prompts)
  - Extract sample frames at key positions
  - Generate static HTML document with embedded images
  - Side-by-side comparison grid for visual evaluation
  - Self-contained (images base64 encoded, no external dependencies)
- [ ] `synth analog` command - Post-processing with grain/color/audio
  - Apply analog effects (grain, breathing, color drift)
  - Apply color grading presets or custom settings
  - Mix in audio tracks with panning

### Phase 6: Next.js UI
- [ ] Basic app shell
- [ ] Video browser component
- [ ] Treatment controls
- [ ] Render API route
- [ ] Progress/queue system

---

## Open Questions

1. **Preset location**: Keep in `presets/` or move to `src/lib/synth/presets/`?
2. **Output naming**: Keep timestamp format or add preset name?
3. **Parallel rendering**: Support queue of multiple renders?
4. **Cloud storage**: Support S3/R2 output in addition to local?

---

## Reference: Current Python → Node.js Mapping

| Python File | Node.js Equivalent |
|-------------|-------------------|
| `blend-video.py` | `colorKeyTreatment` + basic pipeline |
| `blend-video-alt.py` | Full treatment library + composers |
| `blend-video-multi-vid.py` | Audio mixer + looper |
| `upload-video-ci.py` | `cli/commands/upload.ts` |
| Configuration globals | `presets/*.json` + Zod schemas |
| **Post-Blend Pipeline** | |
| `analog-processor.py` | `postprocess/analog.ts` - grain, breathing, color drift |
| `add-random-audio.py` | `postprocess/audio-mixer.ts` - layered audio with panning |
| FFmpeg color filters | `postprocess/color-grade.ts` - presets + custom grading |
| `generative-frame-processor.py` | `ai/frame-processor.ts` - SD/fal.ai integration |
| `sd-grid-sweep.py` | `cli/commands/sweep.ts` - parameter exploration |
| `optical-flow-stabilizer.py` | `postprocess/stabilizer.ts` - temporal smoothing |
| `keyframe-propagation.py` | `ai/keyframe-propagation.ts` - efficient SD with flow |




