from dataclasses import dataclass

@dataclass
class Timm_Checkpoint:
    swin_small_patch4_window7_224: str = "swin_small_patch4_window7_224"
    swin_base_patch4_window7_224: str = "swin_base_patch4_window7_224"
    swin_large_patch4_window7_224: str = "swin_large_patch4_window7_224"
    swin_large_patch4_window12_384: str = "swin_large_patch4_window12_384"

# Example usage
