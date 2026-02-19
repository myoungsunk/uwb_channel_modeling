from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from plots.p0_p13 import _save


def test_save_ignores_pdf_permission_error(monkeypatch, tmp_path: Path):
    fig, _ = plt.subplots()

    original_savefig = fig.savefig

    def _patched_savefig(path, *args, **kwargs):
        if str(path).endswith('.pdf'):
            raise PermissionError('locked file')
        return original_savefig(path, *args, **kwargs)

    monkeypatch.setattr(fig, 'savefig', _patched_savefig)

    out = _save(fig, str(tmp_path), 'P1')
    assert out.endswith('P1.png')
    assert (tmp_path / 'P1.png').exists()
