"""
Code for experiments from the paper:
`Potential of Quantum Machine Learning for Processing Multispectral Earth Observation Data'
This version has been anonymized for the purpose of peer review and may not be used for any other purpose. 
The code for the experiments will be publicly available under an open license - links to the repository 
will be found in the paper.
-------------------------------
RGB data visualisation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def gamma_correction(img, gamma=1 / 2.2):
    """Apply gamma correction for visualization."""
    img = img / np.max(img)
    return np.power(img, gamma)


if __name__ == "__main__":
    PATH = "../data/"
    sdata = np.load(PATH + "sentinel.npz")
    raw_data = sdata["bands"]  # (512, 512, 12)
    raw_gt = sdata["classes"]  # (512, 512)

    classes_names = {
        62: "Artificial surfaces and constructions",
        73: "Cultivated areas",
        82: "Broadleaf tree cover",
        102: "Herbaceous vegetation",
    }

    color_mapping = {
        62: "#FF0000",  # Red
        73: "#FFA500",  # Orange
        82: "#228B22",  # ForestGreen
        102: "#4682B4",  # SteelBlue
    }

    # RGB selection and gamma correction
    rgb_indices = [3, 2, 1]
    rgb_image = gamma_correction(raw_data[..., rgb_indices])

    # Create mask for selected classes
    gt_mask = np.isin(raw_gt, list(classes_names.keys()))

    # Superimposed image
    plt.figure(figsize=(4, 4))
    plt.imshow(rgb_image)
    gt_overlay = np.zeros((*raw_gt.shape, 4))
    for class_val, color in color_mapping.items():
        gt_overlay[raw_gt == class_val] = to_rgba(color, alpha=0.7)
    plt.imshow(gt_overlay)
    patches = [
        mpatches.Patch(color=color_mapping[val], label=val)  # classes_names[val]
        for val in classes_names
    ]
    plt.legend(handles=patches, loc="lower right", framealpha=1)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()

    # Clear RGB image
    plt.figure(figsize=(4, 4))

    plt.imshow(rgb_image)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()

    # GT image on white background with legend
    plt.figure(figsize=(4, 4))
    gt_display = np.ones((*raw_gt.shape, 4))  # white background
    for class_val, color in color_mapping.items():
        gt_display[raw_gt == class_val] = to_rgba(color)
    plt.imshow(gt_display)
    patches = [
        mpatches.Patch(color=color_mapping[val], label=val) for val in classes_names
    ]
    plt.legend(handles=patches, loc="lower right", fontsize=10, framealpha=1.0)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()
