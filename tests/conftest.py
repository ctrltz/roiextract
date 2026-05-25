import mne
import os
import pytest

from mne.datasets import sample


@pytest.fixture(scope="session", autouse=False)
def default_eeg_setup():
    if os.environ.get("BUILD_ENV", "local") == "ci":
        pytest.skip("Skipping EEG setup in CI environment")

    data_path = sample.data_path() / "MEG" / "sample"
    subjects_dir = sample.data_path() / "subjects"
    fwd_path = data_path / "sample_audvis-meg-eeg-oct-6-fwd.fif"
    raw_path = data_path / "sample_audvis_raw.fif"

    # Load the prerequisites: fwd, src, and info
    fwd = mne.read_forward_solution(fwd_path)
    fwd = mne.convert_forward_solution(fwd, force_fixed=True)
    raw = mne.io.read_raw(raw_path)
    raw.set_eeg_reference(projection=True)
    info = raw.info

    # Pick EEG channels only
    eeg_idx = mne.pick_types(info, eeg=True)
    info_eeg = mne.pick_info(info, eeg_idx)
    fwd_eeg = fwd.pick_channels(info_eeg.ch_names)
    raw_eeg = raw.pick_channels(info_eeg.ch_names)

    # Crop 10 seconds to speed up the test
    raw_eeg_crop = raw_eeg.copy().crop(tmax=10.0)

    # Create the inverse operator
    noise_cov = mne.make_ad_hoc_cov(info_eeg, std=1.0)
    inv_op = mne.minimum_norm.make_inverse_operator(
        info_eeg, fwd_eeg, noise_cov, fixed=True, depth=None
    )

    # Load the Desikan-Killiany parcellation labels
    labels = mne.read_labels_from_annot(
        "sample", parc="aparc", subjects_dir=subjects_dir
    )

    return fwd_eeg, inv_op, raw_eeg_crop, labels
