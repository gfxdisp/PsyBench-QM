"""Quick generator for a disc stimulus (glare ring + bandpass noise) and save as PNG.

This is a compact, self-contained script intended for quick previewing.
Usage example:
    python scripts/generate_disc_stimulus.py --out /tmp/stim.png

Dependencies: numpy, pillow
"""
import argparse
import numpy as np
from PIL import Image


def make_bandpass_noise(w, h, noise_freq_cpd, ppd, seed=12345):
    """Generate bandpass-filtered spatial noise (zero-mean, unit std).

    Frequencies are in cycles per degree (cpd). We convert to cycles/pixel
    by dividing by ppd (pixels per degree).
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w))

    # FFT
    F = np.fft.fft2(noise)

    # Frequency coordinates (cycles per pixel)
    fx = np.fft.fftfreq(w)
    fy = np.fft.fftfreq(h)
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX * FX + FY * FY)

    # target frequency in cycles/pixel
    f0 = float(noise_freq_cpd) / float(ppd)
    # band edges: +/- 1 octave (same as MATLAB code using 2^(log2(f)-0.5) .. )
    f_low = 2 ** (np.log2(f0) - 0.5)
    f_high = 2 ** (np.log2(f0) + 0.5)

    mask = (R >= f_low) & (R <= f_high)

    F_filtered = F * mask
    noise_bp = np.fft.ifft2(F_filtered).real

    # normalize to unit std
    s = np.std(noise_bp)
    if s == 0:
        return noise_bp
    return noise_bp / s


def create_stimulus(width=1920, height=1080, ppd=60,
                    noise_radius_deg=5.0, glare_size_deg=0.5, glare_distance_deg=1.3,
                    background_lum=1.0, disk_lum=100.0,
                    noise_freq_cpd=4.0, noise_contrast=0.1, seed=12345):
    """Create a grayscale stimulus image (H x W) in luminance units.

    - background_lum and disk_lum are in arbitrary units (we'll map to 0-255 later).
    - noise_radius_deg and glare_size_deg are specified in degrees and converted to pixels using ppd.
    - glare_distance_deg is distance from noise radius to inner edge of glare ring.
    """
    w = int(width)
    h = int(height)

    # base image
    img = np.ones((h, w), dtype=float) * float(background_lum)

    # radial distance map (pixels)
    ys = np.arange(h) - (h // 2)
    xs = np.arange(w) - (w // 2)
    XX, YY = np.meshgrid(xs, ys)
    Rpx = np.sqrt(XX ** 2 + YY ** 2)

    # compute ring radii in pixels
    noise_radius_px = noise_radius_deg * ppd
    glare_inner = (noise_radius_deg + glare_distance_deg) * ppd
    glare_outer = glare_inner + glare_size_deg * ppd

    # add glare ring (set additive luminance equal to disk_lum in ring)
    ring_mask = (Rpx > glare_inner) & (Rpx <= glare_outer)
    img[ring_mask] += float(disk_lum)
    img = np.clip(img, 0.0, float(disk_lum))

    # generate bandpass noise
    noise = make_bandpass_noise(w, h, noise_freq_cpd, ppd, seed=seed)

    # noise mask (inside noise_radius)
    noise_mask = (Rpx <= noise_radius_px)

    # apply contrast and background scaling (as in MATLAB: noise * noise_contrast * background_lum)
    noise_stim = noise * float(noise_contrast) * float(background_lum)
    img = img + noise_stim * noise_mask
    img = np.clip(img, 0.0, float(disk_lum))

    return img


def save_luminance_png(img, disk_lum, out_path):
    """Map luminance image to 0..255 and save as grayscale PNG.

    This is a fast preview mapping: values are scaled by disk_lum (max). For
    production rendering one would apply proper color encoding / EOTF.
    """
    norm = np.clip(img / float(disk_lum), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    rgb = np.stack([u8, u8, u8], axis=-1)
    im = Image.fromarray(rgb, mode='RGB')
    im.save(out_path)


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--height', type=int, default=768)
    p.add_argument('--ppd', type=float, default=60.0)
    p.add_argument('--noise-radius', type=float, default=2.0, help='degrees')
    p.add_argument('--glare-size', type=float, default=1.0, help='degrees')
    p.add_argument('--glare-distance', type=float, default=2.3, help='degrees')
    p.add_argument('--background-lum', type=float, default=10.0)
    p.add_argument('--disk-lum', type=float, default=100.0)
    p.add_argument('--noise-freq', type=float, default=4.0, help='cycles per degree')
    p.add_argument('--noise-contrast', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    img = create_stimulus(width=args.width, height=args.height, ppd=args.ppd,
                          noise_radius_deg=args.noise_radius, glare_size_deg=args.glare_size,
                          glare_distance_deg=args.glare_distance,
                          background_lum=args.background_lum, disk_lum=args.disk_lum,
                          noise_freq_cpd=args.noise_freq, noise_contrast=args.noise_contrast,
                          seed=args.seed)

    save_luminance_png(img, args.disk_lum, args.out)
    print(f'Wrote {args.out} (shape={img.shape})')


if __name__ == '__main__':
    _cli()
