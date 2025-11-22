# mutag_dataset_compat.py
#
# Stitched-node view of TUDataset("MUTAG") with a MolHIVArchiveDataset-like API.

from proteins_dataset_compat import _TUDStitchedDataset


class MutagArchiveDataset(_TUDStitchedDataset):
    """
    Thin wrapper for MUTAG dataset.
    Accepts the same knobs as MolHIVArchiveDataset, but ignores archive-specific ones.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(name="MUTAG", *args, **kwargs)
