#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

if __name__ == "__main__":
    input_f = Path(sys.argv[1])
    target_i = int(sys.argv[2])
    target_j = int(sys.argv[3])
    target_k = int(sys.argv[4])

    in_shape = np.array(nib.load(input_f).shape[:3])
    target_shape = np.array([target_i, target_j, target_k])

    low = np.floor((target_shape - in_shape) / 2).astype(int).tolist()
    up = np.ceil((target_shape - in_shape) / 2).astype(int).tolist()

    print(
        " ".join(
            [
                "-axis",
                "0",
                f"{low[0]},{up[0]}",
                "-axis",
                "1",
                f"{low[1]},{up[1]}",
                "-axis",
                "2",
                f"{low[2]},{up[2]}",
            ]
        ),
        end="",
    )
