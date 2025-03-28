from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

PSNR = PeakSignalNoiseRatio()
SSIM = StructuralSimilarityIndexMeasure()
