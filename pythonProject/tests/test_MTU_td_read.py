import pytest
import os
import tempfile
import h5py
import numpy as np
from pythonProject.src.utils.MTU_td_read import PhoenixTDReader

def test_merge_24k_from_folder():
    folder = r"F:\Anylysis_project\pythonProject\data\东海\bbmt36\bbmt36\test\10267_2025-10-05-001425\0"
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        out_h5 = tmp.name
    try:
        PhoenixTDReader.merge_24k_from_folder(folder, output_file=out_h5)
        with h5py.File(out_h5, 'r') as hf:
            # 注意组路径
            grp = hf['/24k']
            data = grp['data'][:]
            assert len(data) > 0
            assert grp.attrs['sample_rate_hz'] == 24000
            # 可选的更多断言
    finally:
        os.unlink(out_h5)

def test_merge_150_from_folder():
    folder = r"F:\Anylysis_project\pythonProject\data\东海\bbmt36\bbmt36\test\10267_2025-10-05-001425\0"
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        out_h5 = tmp.name
    try:
        PhoenixTDReader.merge_150_from_folder(folder, output_file=out_h5)
        with h5py.File(out_h5, 'r') as hf:
            grp = hf['/150']
            data = grp['data'][:]
            assert len(data) > 0
            assert grp.attrs['sample_rate_hz'] == 150
    finally:
        os.unlink(out_h5)