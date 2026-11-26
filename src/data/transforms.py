"""
Mathematical and spatial transform specifications for the DFU preprocessing pipeline.
Provides functional definitions for aspect-ratio-preserving letterboxing, interpolation separation,
and signed distance transform precomputation without executing on real datasets.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image


def calculate_letterbox_geometry(
    src_width: int,
    src_height: int,
    target_width: int = 512,
    target_height: int = 512
) -> Dict[str, Any]:
    """
    Computes exact isotropic scaling factors and symmetric padding margins
    to embed an arbitrary aspect ratio image into a target canvas.
    """
    scale = min(target_width / src_width, target_height / src_height)
    new_w = int(round(src_width * scale))
    new_h = int(round(src_height * scale))
    
    pad_left = (target_width - new_w) // 2
    pad_top = (target_height - new_h) // 2
    pad_right = target_width - new_w - pad_left
    pad_bottom = target_height - new_h - pad_top
    
    return {
        "scale": scale,
        "scaled_width": new_w,
        "scaled_height": new_h,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
        "target_width": target_width,
        "target_height": target_height
    }


def apply_letterbox_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (512, 512),
    interpolation_mode: int = Image.BILINEAR,
    fill_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """
    Applies isotropic scaling and padding to an RGB image.
    Uses continuous interpolation (Bilinear or Bicubic).
    """
    w, h = image.size
    geom = calculate_letterbox_geometry(w, h, target_size[0], target_size[1])
    
    resized = image.resize((geom["scaled_width"], geom["scaled_height"]), interpolation_mode)
    new_img = Image.new("RGB", target_size, fill_color)
    new_img.paste(resized, (geom["pad_left"], geom["pad_top"]))
    return new_img


def apply_letterbox_mask(
    mask: Image.Image,
    target_size: Tuple[int, int] = (512, 512),
    fill_value: int = 0
) -> Image.Image:
    """
    Applies isotropic scaling and padding to a segmentation mask.
    MANDATORY: Uses NEAREST-NEIGHBOR interpolation to preserve categorical label integrity.
    """
    w, h = mask.size
    geom = calculate_letterbox_geometry(w, h, target_size[0], target_size[1])
    
    resized = mask.resize((geom["scaled_width"], geom["scaled_height"]), Image.NEAREST)
    new_mask = Image.new("L", target_size, fill_value)
    new_mask.paste(resized, (geom["pad_left"], geom["pad_top"]))
    return new_mask


def compute_tolerance_boundary_fields(
    mask_binary: np.ndarray,
    tau: float = 2.0,
    max_cap: float = 20.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes two-sided tolerance-bounded distance fields (Phi_FP and Phi_FN)
    strictly from the binary ground truth mask.
    Returns (phi_fp, phi_fn) as float32 arrays of shape (H, W).
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt
    
    if mask_binary.sum() == 0:
        # Authentic negative control: zero boundary penalty
        h, w = mask_binary.shape
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    # 1. Morphological 1-pixel boundary
    eroded = binary_erosion(mask_binary, structure=np.ones((3, 3)))
    boundary = mask_binary ^ eroded
    
    # 2. Euclidean distance transform to boundary contour
    edt_boundary = distance_transform_edt(boundary == 0)
    
    # 3. Two-sided distance fields
    # Exterior (False Positive Zone): background pixels (mask == 0) where distance > tau
    phi_fp = np.zeros_like(edt_boundary, dtype=np.float32)
    fp_mask = (mask_binary == 0) & (edt_boundary > tau)
    phi_fp[fp_mask] = np.minimum(edt_boundary[fp_mask] - tau, max_cap)
    
    # Interior (False Negative Zone): foreground pixels (mask == 1) where distance > tau
    phi_fn = np.zeros_like(edt_boundary, dtype=np.float32)
    fn_mask = (mask_binary == 1) & (edt_boundary > tau)
    phi_fn[fn_mask] = np.minimum(edt_boundary[fn_mask] - tau, max_cap)
    
    return phi_fp, phi_fn


# ---------------------------------------------------------------------------
# D3 Augmentation Functions
# ---------------------------------------------------------------------------

def _compose_affine(a1: float, b1: float, c1: float,
                    d1: float, e1: float, f1: float,
                    a2: float, b2: float, c2: float,
                    d2: float, e2: float, f2: float) -> Tuple[float, float, float, float, float, float]:
    """
    Compose two affine transforms: T = T2 ∘ T1 (apply T1 then T2).
    
    T1: x' = a1*x + b1*y + c1,  y' = d1*x + e1*y + f1
    T2: x' = a2*x + b2*y + c2,  y' = d2*x + e2*y + f2
    
    Returns (a, b, c, d, e, f) for the composed transform.
    """
    a = a2 * a1 + b2 * d1
    b = a2 * b1 + b2 * e1
    c = a2 * c1 + b2 * f1 + c2
    d = d2 * a1 + e2 * d1
    e = d2 * b1 + e2 * e1
    f = d2 * c1 + e2 * f1 + f2
    return a, b, c, d, e, f


def _affine_scaling(scale: float, cx: float, cy: float) -> Tuple[float, float, float, float, float, float]:
    """
    Affine transform for scaling by `scale` about centre (cx, cy).
    
    x' = scale*(x - cx) + cx = scale*x + cx*(1-scale)
    y' = scale*(y - cy) + cy = scale*y + cy*(1-scale)
    
    Returns (a, b, c, d, e, f).
    """
    c = cx * (1 - scale)
    f = cy * (1 - scale)
    return scale, 0.0, c, 0.0, scale, f


def _affine_rotation(angle_degrees: float, cx: float, cy: float) -> Tuple[float, float, float, float, float, float]:
    """
    Affine transform for rotation by `angle_degrees` (CCW positive) about centre (cx, cy).
    
    x' = cos(θ)*(x - cx) - sin(θ)*(y - cy) + cx
    y' = sin(θ)*(x - cx) + cos(θ)*(y - cy) + cy
    
    Returns (a, b, c, d, e, f).
    """
    import math
    theta = math.radians(angle_degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    c = cx * (1 - cos_t) + cy * sin_t
    f = cy * (1 - cos_t) - cx * sin_t
    return cos_t, -sin_t, c, sin_t, cos_t, f


def _affine_horizontal_flip(width: int) -> Tuple[float, float, float, float, float, float]:
    """
    Affine transform for horizontal flip: x' = width - 1 - x, y' = y.
    
    Returns (a, b, c, d, e, f).
    """
    return -1.0, 0.0, float(width - 1), 0.0, 1.0, 0.0


def _affine_vertical_flip(height: int) -> Tuple[float, float, float, float, float, float]:
    """
    Affine transform for vertical flip: x' = x, y' = height - 1 - y.
    
    Returns (a, b, c, d, e, f).
    """
    return 1.0, 0.0, 0.0, 0.0, -1.0, float(height - 1)


def apply_affine_transform_image_mask(
    image: Image.Image,
    mask: Image.Image,
    affine: Tuple[float, float, float, float, float, float],
    fill_rgb: Tuple[int, int, int] = (0, 0, 0),
    fill_mask: int = 0,
) -> Tuple[Image.Image, Image.Image]:
    """
    Apply the same affine transform to an RGB image and its corresponding mask,
    using appropriate interpolation for each.
    
    The same affine matrix and coordinate mapping are applied to both RGB and mask,
    ensuring spatial correspondence. RGB uses bilinear interpolation; mask uses
    nearest-neighbour interpolation. Both use constant-zero padding (fill value).
    
    Returns (transformed_image, transformed_mask).
    """
    a, b, c, d, e, f = affine
    
    # Apply the same affine transform to both image and mask
    # Image uses bilinear, mask uses nearest-neighbour
    transformed_image = image.transform(
        image.size,
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.BILINEAR,
        fill=fill_rgb,
    )
    
    transformed_mask = mask.transform(
        mask.size,
        Image.AFFINE,
        (a, b, c, d, e, f),
        resample=Image.NEAREST,
        fill=fill_mask,
    )
    
    return transformed_image, transformed_mask


def apply_brightness(image: np.ndarray, beta: float) -> np.ndarray:
    """
    Apply brightness transformation: multiply all three RGB channels by a shared
    scalar beta.
    
    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3) with values in [0, 1].
    beta : float
        Shared scalar factor ~ Uniform(0.85, 1.15).
    
    Returns
    -------
    np.ndarray
        Brightness-adjusted RGB image.
    """
    result = beta * image
    return result


def apply_contrast(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply contrast transformation: per-image global-mean scaling.
    
    Let mu = mean of all pixels across all channels.
    Then I_out,c,h,w = mu + gamma * (I_in,c,h,w - mu).
    
    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3) with values in [0, 1].
    gamma : float
        Per-image global-mean contrast factor ~ Uniform(0.85, 1.15).
    
    Returns
    -------
    np.ndarray
        Contrast-adjusted RGB image.
    """
    mu = np.mean(image)
    result = mu + gamma * (image - mu)
    return result


def apply_hue_saturation(
    image_rgb: np.ndarray,
    delta_h_degrees: float,
    gamma_sat: float,
) -> np.ndarray:
    """
    Apply hue and saturation transformations in HSV colour space.
    
    Parameters
    ----------
    image_rgb : np.ndarray
        RGB image array of shape (H, W, 3) with values in [0, 1].
    delta_h_degrees : float
        Hue shift in degrees ~ Uniform(-10, +10), wrapped modulo 360.
    gamma_sat : float
        Multiplicative saturation factor ~ Uniform(0.8, 1.2).
    
    Returns
    -------
    np.ndarray
        HSV-transformed RGB image, clamped to [0, 1].
    """
    import numpy as onp
    from PIL import Image
    
    # Convert RGB to HSV using PIL
    # PIL expects uint8 [0,255], so scale up
    img_uint8 = (image_rgb * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode="RGB")
    
    # Convert to HSV
    hsv_img = pil_img.convert("HSV")
    
    # Get pixel data
    pixels = onp.array(hsv_img.getdata()).reshape(-1, 3)
    h = pixels[:, 0].reshape(img_uint8.shape[:2])
    s = pixels[:, 1].reshape(img_uint8.shape[:2])
    v = pixels[:, 2].reshape(img_uint8.shape[:2])
    
    # Apply hue shift: delta_H in degrees, wrapped modulo 360
    # PIL HSV hue is in [0, 255] representing [0, 360] degrees
    hue_scale = 255.0 / 360.0
    h_new = (h.astype(float) + delta_h_degrees * hue_scale) % 255
    h_new = onp.clip(h_new, 0, 255).astype(np.uint8)
    
    # Apply multiplicative saturation: S_out = S_in * gamma, clamped to [0, 255]
    s_new = (s.astype(float) * gamma_sat).clip(0, 255).astype(np.uint8)
    
    # V channel unchanged
    v_new = v.astype(np.uint8)
    
    # Reconstruct HSV image
    reconstructed = onp.stack([h_new, s_new, v_new], axis=-1).astype(np.uint8)
    result_img = Image.fromarray(reconstructed, mode="HSV")
    
    # Convert back to RGB
    rgb_img = result_img.convert("RGB")
    rgb_array = np.array(rgb_img, dtype=np.float32) / 255.0
    
    # Final clamp to [0, 1]
    rgb_array = np.clip(rgb_array, 0, 1)
    
    return rgb_array


def apply_gaussian_blur(
    image: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Apply Gaussian blur to an RGB image.
    
    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3) with values in [0, 1].
    sigma : float
        Gaussian sigma ~ Uniform(0.1, 1.5).
    
    Returns
    -------
    np.ndarray
        Blurred RGB image with values in [0, 1].
    """
    import scipy.ndimage
    
    # Kernel size: k = 2 * ceil(3*sigma) + 1 (odd integer)
    k = 2 * int(np.ceil(3 * sigma)) + 1
    k = max(k, 1)  # ensure at least 1
    
    # Apply Gaussian filter with zero (constant) padding to all 3 channels
    # scipy.ndimage.gaussian_filter with mode='constant' and cval=0 does zero-padding
    result = scipy.ndimage.gaussian_filter(
        image,
        sigma=sigma,
        mode="constant",
        cval=0.0,
    )
    
    # Clip to [0, 1] to maintain contract
    result = np.clip(result, 0, 1)
    
    return result


def apply_cutout(
    image: np.ndarray,
    mask: np.ndarray,
    area_min: float = 0.05,
    area_max: float = 0.15,
    aspect_ratio_min: float = 0.5,
    aspect_ratio_max: float = 2.0,
    max_attempts: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Random CutOut to an RGB image.
    
    The cutout rectangle is applied to the RGB image only; the mask is never
    modified. A safety test ensures the rectangle contains no foreground mask
    pixels (value=1). If any foreground pixel is inside, the centre is redrawn
    (max max_attempts). If no safe centre after max_attempts, CutOut is skipped.
    
    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3) with values in [0, 1].
    mask : np.ndarray
        Binary mask array of shape (H, W) with values in {0, 1}.
    area_min : float
        Minimum area as fraction of canvas (5%).
    area_max : float
        Maximum area as fraction of canvas (15%).
    aspect_ratio_min : float
        Minimum aspect ratio (0.5).
    aspect_ratio_max : float
        Maximum aspect ratio (2.0).
    max_attempts : int
        Maximum centre redraw attempts (10).
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (augmented_image, mask_unchanged) where mask is never modified.
    """
    H, W = image.shape[:2]
    
    # Compute area and aspect ratio
    area = np.random.uniform(area_min, area_max) * (H * W)
    aspect_ratio = np.random.uniform(aspect_ratio_min, aspect_ratio_max)
    
    # Compute width and height from area and aspect ratio
    # area = w * h, aspect_ratio = w / h  =>  w = sqrt(area * aspect_ratio), h = sqrt(area / aspect_ratio)
    w = int(np.round(np.sqrt(area * aspect_ratio)))
    h = int(np.round(np.sqrt(area / aspect_ratio)))
    
    # Ensure w and h are at least 1 and at most image dimensions
    w = min(w, W)
    h = min(h, H)
    w = max(w, 1)
    h = max(h, 1)
    
    # Centre must keep rectangle fully within the 512×512 image
    # c_x ∈ [w/2, 512−w/2], c_y ∈ [h/2, 512−h/2]
    c_x_min = w // 2
    c_x_max = W - (w // 2) - 1  # -1 to be strict about "fully within"
    c_y_min = h // 2
    c_y_max = H - (h // 2) - 1
    
    # If the rectangle can't fit, skip
    if c_x_min > c_x_max or c_y_min > c_y_max:
        return image, mask
    
    # Try to find a safe centre (rectangle with no foreground mask pixels)
    for attempt in range(max_attempts):
        c_x = int(np.random.randint(c_x_min, c_x_max + 1))
        c_y = int(np.random.randint(c_y_min, c_y_max + 1))
        
        # Check rectangle bounds
        x1 = c_x - w // 2
        y1 = c_y - h // 2
        x2 = x1 + w
        y2 = y1 + h
        
        # Clamp to image boundaries
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, W)
        y2 = min(y2, H)
        
        # Intersect rectangle with mask; if any foreground pixel (value=1) inside, reject
        mask_roi = mask[y1:y2, x1:x2]
        if mask_roi.sum() == 0:
            # Safe: no foreground pixels inside the rectangle
            # Apply CutOut to RGB only
            image_copy = image.copy()
            image_copy[y1:y2, x1:x2] = 0.0  # constant-zero background
            return image_copy, mask
    
    # After max_attempts, skip CutOut entirely
    return image, mask


def augment_sample(
    image: np.ndarray,
    mask: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the full D3 augmentation pipeline to one training sample.
    
    The pipeline applies eight independent activation decisions with approved
    probabilities.
    
    Activation order (as approved):
    1. Geometric transforms (flip, rotation, scaling) - composed into one affine matrix
       via independent Bernoulli draws: hflip p=0.5, vflip p=0.5, rotation p=0.5,
       scaling p=0.3 (factor ∼ Uniform[0.9, 1.1])
    2. Brightness-Contrast family (p=0.4 single Bernoulli draw)
       - When activated, both brightness (shared scalar β ∼ Uniform(0.85, 1.15))
         and contrast (γ ∼ Uniform(0.85, 1.15) about per-image global mean) are applied.
       - Order: brightness then contrast
    3. Hue-Saturation family (p=0.4 independent Bernoulli draw)
       - When activated, both hue (ΔH ∼ Uniform(−10, +10) degrees, HSV, wrapped modulo 360°)
         and saturation (multiplicative S_out = S_in · γ, γ ∼ Uniform(0.8, 1.2)) are applied.
       - Order: after brightness/contrast, before Gaussian blur
    4. Gaussian blur (p=0.2)
       - σ ∼ Uniform(0.1, 1.5); kernel k = 2⌈3σ⌉ + 1 (odd integer);
         zero (constant) boundary padding; applied identically to all 3 RGB channels.
       - Placement: after brightness, contrast, hue/saturation, and before CutOut.
    5. CutOut (p=0.3)
       - Probability p = 0.3 per sample. Area: Uniform(5%, 15%) of canvas.
         Aspect ratio: Uniform(0.5, 2.0). Rectangle must remain entirely within
         the 512×512 image using actual width w and height h.
         Centre drawn from region keeping rectangle fully inside: c_x ∈ [w/2, 512−w/2],
         c_y ∈ [h/2, 512−h/2]. Safety test: intersect rectangle with current binary mask;
         if any foreground pixel (value=1) inside, reject and redraw centre (max 10 attempts).
         If no safe centre after 10 attempts, skip CutOut for this sample (no CutOut applied).
         Applied to RGB only; mask never modified. No lesion-area safety margin.
    
    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3) with values in [0, 1].
    mask : np.ndarray
        Binary mask array of shape (H, W) with values in {0, 1}.
    seed : Optional[int]
        Seed for reproducibility. If provided, sets numpy random state.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (augmented_image, augmented_mask) where both are in the correct contract:
        image in [0, 1], mask in {0, 1}.
    """
    import numpy as onp
    import math
    
    if seed is not None:
        onp.random.seed(seed)
    
    H, W = image.shape[:2]
    cx, cy = W / 2, H / 2  # image centre for affine transforms
    
    # ------------------------------------------------------------------
    # D3-d: Geometric transforms composed into one affine matrix
    # Four independent Bernoulli draws:
    #   Horizontal flip p=0.5
    #   Vertical flip p=0.5
    #   Rotation p=0.5 (angle ∼ Uniform[-15°, +15°], CCW positive)
    #   Scaling p=0.3 (factor ∼ Uniform[0.9, 1.1])
    # ------------------------------------------------------------------
    
    # Start with identity affine: (a, b, c, d, e, f) where
    # x' = a*x + b*y + c,  y' = d*x + e*y + f
    # Identity: a=1, b=0, c=0, d=0, e=1, f=0
    affine: Tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    
    # Horizontal flip (p=0.5)
    if onp.random.random() < 0.5:
        affine = _compose_affine(
            *affine, * _affine_horizontal_flip(W)
        )
    
    # Vertical flip (p=0.5)
    if onp.random.random() < 0.5:
        affine = _compose_affine(
            *affine, * _affine_vertical_flip(H)
        )
    
    # Rotation (p=0.5)
    if onp.random.random() < 0.5:
        angle = onp.random.uniform(-15, 15)  # Uniform [-15°, +15°]
        affine = _compose_affine(
            *affine, * _affine_rotation(angle, cx, cy)
        )
    
    # Scaling (p=0.3)
    if onp.random.random() < 0.3:
        scale = onp.random.uniform(0.9, 1.1)  # Uniform [0.9, 1.1]
        affine = _compose_affine(
            *affine, * _affine_scaling(scale, cx, cy)
        )
    
    # Apply the composed affine transform to both image and mask
    pil_image = Image.fromarray((image * 255).astype(np.uint8), mode="RGB")
    pil_mask = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    
    pil_image, pil_mask = apply_affine_transform_image_mask(
        pil_image, pil_mask, affine,
        fill_rgb=(0, 0, 0),
        fill_mask=0,
    )
    
    image = np.array(pil_image, dtype=np.float32) / 255.0
    mask = (np.array(pil_mask, dtype=np.float32) / 255.0).squeeze()
    
    # ------------------------------------------------------------------
    # D3-e: Brightness-Contrast family (p=0.4 single Bernoulli draw)
    # ------------------------------------------------------------------
    if onp.random.random() < 0.4:
        # When activated, both brightness and contrast are applied
        # Brightness: shared scalar beta ∼ Uniform(0.85, 1.15)
        beta = onp.random.uniform(0.85, 1.15)
        image = apply_brightness(image, beta)
        
        # Contrast: per-image global-mean gamma ∼ Uniform(0.85, 1.15)
        gamma = onp.random.uniform(0.85, 1.15)
        image = apply_contrast(image, gamma)
    
    # ------------------------------------------------------------------
    # D3-f: Hue-Saturation family (p=0.4 independent family draw)
    # ------------------------------------------------------------------
    if onp.random.random() < 0.4:
        # When activated, both hue and saturation are applied
        delta_h = onp.random.uniform(-10, 10)  # degrees
        gamma_sat = onp.random.uniform(0.8, 1.2)  # multiplicative saturation
        image = apply_hue_saturation(image, delta_h, gamma_sat)
    
    # ------------------------------------------------------------------
    # D3-g: Gaussian blur (p=0.2)
    # ------------------------------------------------------------------
    if onp.random.random() < 0.2:
        sigma = onp.random.uniform(0.1, 1.5)
        image = apply_gaussian_blur(image, sigma)
    
    # ------------------------------------------------------------------
    # D3-h: CutOut (p=0.3)
    # ------------------------------------------------------------------
    if onp.random.random() < 0.3:
        image, mask = apply_cutout(image, mask)
    
    # ------------------------------------------------------------------
    # D3-i: Final RGB clamping and mask binarization
    # ------------------------------------------------------------------
    image = np.clip(image, 0, 1)
    mask = np.clip(mask, 0, 1)
    mask = (mask > 0.5).astype(np.float32)  # ensure binary {0, 1}
    
    return image, mask


def derive_augmentation_seed(training_seed: int, record_id: str, epoch: int) -> int:
    """Derive a deterministic D3 augmentation seed.

    Approved seed policy (G4 D3 amendment): the augmentation seed is derived
    from the configured training seed, the stable record identifier, and the
    current training epoch. It is intentionally independent of shuffled dataset
    positional indices.

    Parameters
    ----------
    training_seed:
        Configured deterministic training seed for the run.
    record_id:
        Stable source-training record identifier (for example,
        ``source_training_0001``).
    epoch:
        Current training epoch.

    Returns
    -------
    int
        Deterministic augmentation seed.
    """
    import hashlib

    digest = hashlib.sha256(
        f"{int(training_seed)}:{str(record_id)}:{int(epoch)}".encode("utf-8")
    ).hexdigest()
    return int(digest, 16) & 0xFFFFFFFF