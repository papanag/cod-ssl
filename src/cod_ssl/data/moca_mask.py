from cod_ssl.data.video_manifest import ManifestVideoCODDataset


class MoCAMaskDenseDataset(ManifestVideoCODDataset):
    """Verified Original-MoCA context with official MoCA-Mask manual targets."""


class MoCAMaskPublicSparseDataset(ManifestVideoCODDataset):
    """Explicit legacy adapter for the 4,691-pair public sparse release."""


MoCAMaskDataset = MoCAMaskPublicSparseDataset
