# How It Works

A technical deep-dive into the Richard Cigarette video synthesis pipeline.

---

## The Core Concept: Colorkey Compositing

The fundamental technique is **colorkey** (also called chroma key or color keying) — the same technology used for green screens in movies.

### What is Colorkey?

Colorkey makes pixels of a specific color transparent:

```
Original pixel: RGB(255, 255, 255)  →  White
Colorkey: 0xFFFFFF (white)
Result: That pixel becomes transparent
```

When you stack videos with transparent pixels on top of each other, you see through to the layers below — creating a composite image.

### The Artistic Twist

Instead of using a clean green/blue screen like film studios, we're keying out colors that naturally occur in the videos:

- **White** (`0xFFFFFF`) — skies, highlights, overexposed areas
- **Black** (`0x000000`) — shadows, dark areas, night scenes
- **Any color** — the ML version extracts actual dominant colors

This creates unpredictable, organic transparency patterns that reveal layers beneath.

---

## The FFMPEG Pipeline

All scripts use FFMPEG, a powerful command-line video processor. Here's how a typical filter chain works:

### Input Stage

```bash
ffmpeg -i video1.mp4 -i video2.mp4 -i video3.mp4
```

Each input gets an index: `[0:v]`, `[1:v]`, `[2:v]` for video streams.

### Filter Complex

The `-filter_complex` flag chains multiple operations together:

```
[0:v]scale=800x800[base];
[1:v]scale=800x800,colorkey=0xffffff:0.3:0.1[ck1];
[base][ck1]overlay=shortest=1[out]
```

Let's break this down:

#### 1. Scale the Base Layer
```
[0:v]scale=800x800[base]
```
- `[0:v]` — Take video stream from first input
- `scale=800x800` — Resize to 800×800 pixels
- `[base]` — Name this output "base" for later reference

#### 2. Process the Top Layer
```
[1:v]scale=800x800,colorkey=0xffffff:0.3:0.1[ck1]
```
- `scale=800x800` — Match dimensions with base
- `colorkey=0xffffff:0.3:0.1` — Make white transparent
  - `0xffffff` — Target color (white)
  - `0.3` — Similarity threshold (how close a color must be)
  - `0.1` — Blend factor (edge softness)
- `[ck1]` — Name this output "ck1"

#### 3. Overlay
```
[base][ck1]overlay=shortest=1[out]
```
- `[base][ck1]` — Take both prepared streams
- `overlay` — Stack ck1 on top of base
- `shortest=1` — End when the shortest input ends

---

## The Scripts in Detail

### blend-video.py — The Simplest Version

```
┌─────────────┐
│  Video 1    │ ──scale──────────────────────────────┐
└─────────────┘                                      │
                                                     ▼
┌─────────────┐                               ┌──────────────┐
│  Video 2    │ ──scale──colorkey──────────── │   Overlay    │ ──► Output
└─────────────┘                               └──────────────┘
                                                     ▲
┌─────────────┐                                      │
│  Video 3    │ ──scale──colorkey────────────────────┘
└─────────────┘
```

1. First video is the base (no colorkey)
2. Subsequent videos get colorkey applied
3. All are overlaid in sequence

### blend-video-alt.py — Random Start Times

Adds sophistication with **trim** operations:

```
[0:v]trim=start=47:duration=30,setpts=PTS-STARTPTS
```

- `trim=start=47` — Start at the 47-second mark
- `duration=30` — Take 30 seconds of footage
- `setpts=PTS-STARTPTS` — Reset timestamps (required after trim)

This means each run gives you a different segment of each video.

### blend-video-multi-vid.py — Looping Short Clips

Handles videos shorter than the desired output:

```
loop=loop=-1:size=1080,setpts=N/(30*TB)
```

- `loop=-1` — Loop infinitely
- `size=1080` — Buffer size (frames to loop)
- The outer `-t` flag cuts to final duration

This ensures even 5-second clips can be used for 2-minute outputs.

### blend-video-alt-color-key.py — Machine Learning

The most sophisticated version uses **K-Means clustering** to find dominant colors.

#### K-Means Explained

K-Means is a clustering algorithm that groups similar data points:

```
Imagine throwing 10,000 pixels into a 3D space (R, G, B axes).
K-Means finds the N most crowded regions — the dominant colors.
```

**Algorithm:**
1. Randomly place N cluster centers
2. Assign each pixel to its nearest center
3. Move each center to the average position of its pixels
4. Repeat until centers stop moving

**Result:** N representative colors from the image

#### The Process

```
Video → Extract Frame → K-Means → Select Colors → Build Colorkey Filter
```

1. **Extract Frame:** Pull a random frame from the video
2. **Downsample:** Resize to 150×150 for speed
3. **K-Means:** Find 4 dominant colors
4. **Select:** Randomly pick 2 colors to key out
5. **Filter:** Build `colorkey=0x{color}:0.3:0.2` for each

---

## Audio Pipeline

Audio is handled separately from video:

### Single Audio Source
```
[2:a]aloop=loop=-1:size=2e+09,asetpts=N/SR/TB[audio]
```
- `aloop` — Loop the audio to fill the video duration
- `size=2e+09` — Large buffer for seamless looping
- `asetpts` — Reset audio timestamps

### Multiple Audio Sources
```
[0:a]atrim=start=12:duration=60...[a0];
[1:a]atrim=start=8:duration=60...[a1];
[a0][a1]amix=inputs=2:normalize=1[outa]
```
- `atrim` — Trim audio to match video segment
- `amix` — Mix multiple audio tracks together
- `normalize=1` — Prevent clipping by auto-adjusting volume

---

## The Overlay Chain

For N videos, the overlay builds a chain:

```
N=2:  [v0][v1]overlay[out]

N=3:  [v0][v1]overlay[temp1];
      [temp1][v2]overlay[out]

N=5:  [v0][v1]overlay[temp1];
      [temp1][v2]overlay[temp2];
      [temp2][v3]overlay[temp3];
      [temp3][v4]overlay[out]
```

Each step composites one more layer onto the stack.

---

## Colorkey Parameters Explained

```
colorkey=color:similarity:blend
```

### color (e.g., `0xffffff`)
The exact color to target. Common choices:
- `0xffffff` — White
- `0x000000` — Black
- `0xff0000` — Red

### similarity (0.0 - 1.0)
How close a pixel's color must be to match:
- `0.1` — Very strict, only near-exact matches
- `0.3` — Moderate, catches similar shades (recommended)
- `0.5` — Loose, removes wide range of similar colors

### blend (0.0 - 1.0)
Edge softness:
- `0.0` — Hard edges (pixelated, digital look)
- `0.1` — Slight softness (recommended)
- `0.5` — Very soft, gradient edges

---

## Debug Output

The ML script (`blend-video-alt-color-key.py`) creates debug images:

```
output_2024-12-02--14-30-00.mp4
output_2024-12-02--14-30-00--debug.png
```

The debug PNG shows:
- Each video's extracted color palette
- The filename of each source video

This helps you understand why certain areas became transparent.

---

## Performance Considerations

### Encoding Settings

```python
'-c:v', 'libx264',  # H.264 codec (widely compatible)
'-crf', '23',       # Quality (18=high, 23=medium, 28=low)
'-preset', 'veryfast'  # Speed vs size tradeoff
```

- Lower CRF = better quality, larger files
- Faster preset = quicker encoding, larger files

### Frame Rate

```python
FRAME_RATE = 18
```

- Lower FPS = dreamier, more abstract feel
- Higher FPS = smoother motion
- 18-24 is a good range for experimental video art

### Resolution

```python
OUTPUT_SIZE = "800x800"
```

- Square formats work well for social media
- Higher resolution = longer processing time
- 800×800 balances quality and speed

---

## Troubleshooting

### "No audio stream found"
Some source videos don't have audio. The scripts handle this gracefully by checking for audio streams before processing.

### "Not enough videos"
Add more source videos to `input/video`. The multi-vid script needs at least 5 by default.

### Output looks too transparent
Lower the `SIMILARITY` value. Try `0.15` instead of `0.3`.

### Output looks barely affected
Raise the `SIMILARITY` value, or switch which colors you're keying out.

### Processing is slow
- Lower the `OUTPUT_SIZE`
- Reduce the number of input videos
- Use a faster encoding preset

---

## Further Reading

- [FFMPEG Filter Documentation](https://ffmpeg.org/ffmpeg-filters.html)
- [Colorkey Filter Reference](https://ffmpeg.org/ffmpeg-filters.html#colorkey)
- [K-Means Clustering Explained](https://scikit-learn.org/stable/modules/clustering.html#k-means)

