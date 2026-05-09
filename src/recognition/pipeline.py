"""
pipeline.py
===========

End-to-end ALPR orchestration:

    preprocess -> detect plate -> normalize -> segment characters
    -> extract features -> classify -> postprocess

The pipeline deliberately accepts a classifier object instead of hiding
training inside recognition.  Any classifier with ``predict_proba`` and
``classes_`` (such as our KNN or MLP) can be plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.detection import DetectionConfig, DetectionResult, detect_plate
from src.features import FeatureConfig, extract_batch_features
from src.normalization import NormalizationConfig, NormalizationResult, normalize_plate
from src.preprocessing import PreprocessConfig, PreprocessResult, preprocess
from src.recognition.postprocessing import PostprocessResult, postprocess_predictions
from src.segmentation import SegmentationConfig, SegmentationResult, segment_characters


@dataclass
class RecognitionConfig:
    """Configuration bundle for the full recognition pipeline."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    min_confidence: float = 0.50
    use_vietnam_format: bool = True


@dataclass
class RecognitionResult:
    """Structured output of :func:`recognize_license_plate`."""

    text: str
    raw_text: str
    corrected_text: str
    confidences: list[float]
    preprocess: PreprocessResult
    detection: DetectionResult
    normalization: Optional[NormalizationResult]
    segmentation: Optional[SegmentationResult]
    postprocess: PostprocessResult

    @property
    def found_plate(self) -> bool:
        return bool(self.detection.candidates)

    @property
    def found_characters(self) -> bool:
        return self.segmentation is not None and bool(self.segmentation.characters)


def recognize_license_plate(
    image: np.ndarray,
    classifier: Any,
    config: Optional[RecognitionConfig] = None,
) -> RecognitionResult:
    """
    Run the full license-plate recognition pipeline.

    Parameters
    ----------
    image : np.ndarray
        RGB or grayscale input image.
    classifier : object
        Fitted classifier with a ``predict_proba(X)`` method and a
        ``classes_`` array.  The KNN and MLP modules both satisfy this.
    config : RecognitionConfig, optional
        Parameters for every pipeline stage.

    Returns
    -------
    RecognitionResult
        Final text plus all major intermediate stages for debugging.
    """
    cfg = config or RecognitionConfig()

    pre = preprocess(image, cfg.preprocess)
    det = detect_plate(pre.enhanced, cfg.detection)
    empty_post = postprocess_predictions([], min_confidence=cfg.min_confidence)

    if not det.candidates:
        return RecognitionResult(
            text="",
            raw_text="",
            corrected_text="",
            confidences=[],
            preprocess=pre,
            detection=det,
            normalization=None,
            segmentation=None,
            postprocess=empty_post,
        )

    norm = normalize_plate(pre.enhanced, det.candidates[0], cfg.normalization)
    seg = segment_characters(norm.normalized, cfg.segmentation)
    if not seg.characters:
        return RecognitionResult(
            text="",
            raw_text="",
            corrected_text="",
            confidences=[],
            preprocess=pre,
            detection=det,
            normalization=norm,
            segmentation=seg,
            postprocess=empty_post,
        )

    X = extract_batch_features((char.normalized for char in seg.characters), cfg.features)
    labels, confidences = _predict_labels_and_confidences(classifier, X)
    post = postprocess_predictions(
        labels,
        confidences,
        min_confidence=cfg.min_confidence,
        use_vietnam_format=cfg.use_vietnam_format,
    )

    return RecognitionResult(
        text=post.formatted_text,
        raw_text=post.raw_text,
        corrected_text=post.corrected_text,
        confidences=confidences,
        preprocess=pre,
        detection=det,
        normalization=norm,
        segmentation=seg,
        postprocess=post,
    )


def _predict_labels_and_confidences(
    classifier: Any,
    X: np.ndarray,
) -> tuple[list[str], list[float]]:
    """Use a fitted classifier to get labels and max probabilities."""
    if not hasattr(classifier, "predict_proba"):
        raise ValueError("classifier must provide a predict_proba(X) method.")
    if not hasattr(classifier, "classes_"):
        raise ValueError("classifier must expose a classes_ array.")

    classes = np.asarray(classifier.classes_).astype(str)
    proba = np.asarray(classifier.predict_proba(X), dtype=np.float32)
    if proba.ndim != 2 or proba.shape[0] != X.shape[0]:
        raise ValueError(
            "classifier.predict_proba(X) must return an (N, C) probability matrix."
        )
    if proba.shape[1] != classes.shape[0]:
        raise ValueError(
            f"Probability columns ({proba.shape[1]}) do not match classes ({classes.shape[0]})."
        )

    idx = np.argmax(proba, axis=1)
    labels = classes[idx].tolist()
    confidences = np.max(proba, axis=1).astype(float).tolist()
    return labels, confidences
