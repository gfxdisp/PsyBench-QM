"""
Grid preview video for DiskTemporalFreqAchTest.
  columns (x-axis) : temporal frequencies  (low → high)
  rows    (y-axis) : contrast levels        (high → low, top to bottom)
Output: figs/disk_temporal_freq_ach_grid.mp4
"""

import numpy as np
import torch
import os

from PIL import Image, ImageDraw, ImageFont

from pyst.GaborPatch import DiskTemporalFreqAchTest
from pyst.utils import tensor_to_numpy_image
from pycvvdp.video_writer import VideoWriter


# ── Grid parameters ────────────────────────────────────────────────────────────
N_CONTRASTS = 10          # rows     (y-axis)
N_FREQ      = 10          # columns  (x-axis)
CELL_PX     = 80          # each cell scaled to this size in the composite (px)
BORDER      = 4           # white gap between cells (px)
PAD_L       = 120         # left margin for y-axis labels
PAD_T       = 90          # top  margin for x-axis labels

OUTPUT = os.path.join('figs', 'disk_temporal_freq_ach_grid.mp4')

_FONT_REGULAR = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
_FONT_BOLD    = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


# ── Instantiate test ───────────────────────────────────────────────────────────
device = torch.device('cpu')
Test   = DiskTemporalFreqAchTest(N_contrasts=N_CONTRASTS, N_freq=N_FREQ, device=device)

N_FRAMES  = Test.frames
FPS       = Test.fps
contrasts = Test.contrasts.cpu().numpy()            # (N_CONTRASTS,)
t_freqs   = Test.temporal_frequencies.cpu().numpy() # (N_FREQ,)


# ── Pre-cache all cell frames as uint8 ────────────────────────────────────────
# Shape: (N_CONTRASTS, N_FREQ, N_FRAMES, CELL_PX, CELL_PX, 3)
print("Pre-generating cell frames...")
cell_cache = np.empty(
    (N_CONTRASTS, N_FREQ, N_FRAMES, CELL_PX, CELL_PX, 3), dtype=np.uint8
)

# meshgrid is ij-indexed → contrast is outer loop, t_freq is inner
for c_idx in range(N_CONTRASTS):
    for tf_idx in range(N_FREQ):
        index   = c_idx * N_FREQ + tf_idx
        vid_src, _dm, _gm = Test.get_condition(index)

        for ff in range(N_FRAMES):
            frame_f = tensor_to_numpy_image(
                vid_src.get_test_frame(ff, colorspace='display_encoded_dmax', device=device)
            )  # H×W×3 float32 [0,1]
            frame_u8 = (frame_f * 255).clip(0, 255).astype(np.uint8)
            cell_img = Image.fromarray(frame_u8).resize((CELL_PX, CELL_PX), Image.LANCZOS)
            cell_cache[c_idx, tf_idx, ff] = np.asarray(cell_img)

        del vid_src   # free the ~188 MB tensor immediately
        print(f"  contrast={contrasts[c_idx]:.3f}  t_freq={t_freqs[tf_idx]:.1f} Hz")


# ── Layout helpers ─────────────────────────────────────────────────────────────
GRID_W = PAD_L + BORDER + N_FREQ       * (CELL_PX + BORDER)
GRID_H = PAD_T + BORDER + N_CONTRASTS  * (CELL_PX + BORDER)


def cell_xy(col, vrow):
    """Top-left pixel of cell at grid column `col`, visual row `vrow`."""
    x = PAD_L + BORDER + col  * (CELL_PX + BORDER)
    y = PAD_T + BORDER + vrow * (CELL_PX + BORDER)
    return x, y


def visual_col(tf_idx):
    """Temporal frequency index → column: low freq at left (col 0)."""
    return tf_idx


def visual_row(c_idx):
    """Contrast index → visual row: highest contrast at top (row 0)."""
    return N_CONTRASTS - 1 - c_idx


# ── Font loading ───────────────────────────────────────────────────────────────
def _load_font(path, size):
    if os.path.exists(path):
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()

fnt_val  = _load_font(_FONT_REGULAR, 18)   # axis tick labels
fnt_lbl  = _load_font(_FONT_BOLD,    22)   # axis titles


def text_wh(draw, text, font):
    try:
        bb = draw.textbbox((0, 0), text, font=font)
        return bb[2] - bb[0], bb[3] - bb[1]
    except AttributeError:
        return len(text) * 10, 16


# ── Build static background (titles, arrows, tick labels) ─────────────────────
bg   = Image.new('RGB', (GRID_W, GRID_H), (255, 255, 255))
draw = ImageDraw.Draw(bg)

# ── x-axis (top): "Temporal Frequency (Hz)" title + right arrow + tick labels ──
title_x = 'Temporal Frequency (Hz)'
tw, th  = text_wh(draw, title_x, fnt_lbl)
draw.text((GRID_W // 2 - tw // 2, 4), title_x, fill=(0, 0, 0), font=fnt_lbl)

arrow_y  = 4 + th + 8
arrow_x0 = PAD_L + BORDER
arrow_x1 = GRID_W - BORDER - 10
draw.line([(arrow_x0, arrow_y), (arrow_x1, arrow_y)], fill=(0, 0, 0), width=3)
draw.polygon(
    [(arrow_x1, arrow_y - 6), (arrow_x1, arrow_y + 6), (arrow_x1 + 10, arrow_y)],
    fill=(0, 0, 0),
)

for tf_idx in range(N_FREQ):
    cx, _ = cell_xy(visual_col(tf_idx), 0)
    cx   += CELL_PX // 2
    lbl   = f'{t_freqs[tf_idx]:.1f}'
    tw, th = text_wh(draw, lbl, fnt_val)
    draw.text((cx - tw // 2, PAD_T - th - 5), lbl, fill=(0, 0, 0), font=fnt_val)

# ── y-axis (left): "Contrast" title + upward arrow + tick labels ──────────────
title_y = 'Contrast'
# Use textbbox to get exact glyph bounds and avoid clipping
bb_y    = draw.textbbox((0, 0), title_y, font=fnt_lbl)
tw, th  = bb_y[2] - bb_y[0], bb_y[3] - bb_y[1]
VPAD    = 10   # extra padding so no glyph gets clipped

# Render with bbox-offset correction so glyphs aren't truncated
tmp_title = Image.new('RGB', (tw + 8, th + VPAD * 2), (255, 255, 255))
ImageDraw.Draw(tmp_title).text((4 - bb_y[0], VPAD - bb_y[1]), title_y, fill=(0, 0, 0), font=fnt_lbl)
tmp_title = tmp_title.rotate(90, expand=True)   # size becomes (th+2*VPAD, tw+8)
paste_ty  = GRID_H // 2 - tmp_title.size[1] // 2
bg.paste(tmp_title, (4, paste_ty))

# Upward arrow to the right of the title (small contrast at bottom → high at top)
arrow_x  = 4 + tmp_title.size[0] + 8
arrow_y0 = GRID_H - BORDER         # bottom  (low contrast)
arrow_y1 = PAD_T + BORDER + 10     # top     (high contrast)
draw.line([(arrow_x, arrow_y0), (arrow_x, arrow_y1)], fill=(0, 0, 0), width=3)
draw.polygon(
    [(arrow_x - 6, arrow_y1), (arrow_x + 6, arrow_y1), (arrow_x, arrow_y1 - 10)],
    fill=(0, 0, 0),
)

# Contrast tick labels to the left of cells
for c_idx in range(N_CONTRASTS):
    _, cy = cell_xy(0, visual_row(c_idx))
    cy   += CELL_PX // 2
    lbl   = f'{contrasts[c_idx]:.3f}'
    tw, th = text_wh(draw, lbl, fnt_val)
    draw.text((PAD_L - tw - 10, cy - th // 2), lbl, fill=(0, 0, 0), font=fnt_val)


# ── Composite and write video ──────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
vw = VideoWriter(OUTPUT, fps=FPS, codec='h265')

print(f"\nCompositing {N_FRAMES} frames into {GRID_W}×{GRID_H} grid video...")
for ff in range(N_FRAMES):
    canvas = bg.copy()

    for c_idx in range(N_CONTRASTS):
        for tf_idx in range(N_FREQ):
            x, y = cell_xy(visual_col(tf_idx), visual_row(c_idx))
            canvas.paste(Image.fromarray(cell_cache[c_idx, tf_idx, ff]), (x, y))

    vw.write_frame_rgb(np.asarray(canvas).astype(np.float32) / 255.0)

    if ff % 20 == 0:
        print(f"  Frame {ff + 1}/{N_FRAMES}")

vw.close()
print(f"\nSaved: {OUTPUT}")
