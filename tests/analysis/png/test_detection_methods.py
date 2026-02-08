import pytest
import pickle

from hsnn import utils
from hsnn.analysis import ResultsDatabase
from hsnn.analysis.png import detection, refinery

SAMPLES_DIR = utils.BASE_DIR / "tests/data/detection"


@pytest.fixture(scope="module")
def real_data():
    """Loads the real data samples extracted from the notebook."""
    records_path = SAMPLES_DIR / "records_sample.pkl"
    syn_params_path = SAMPLES_DIR / "syn_params_sample.pkl"
    labels_path = SAMPLES_DIR / "labels_sample.pkl"

    if not records_path.exists():
        pytest.skip(f"Real data samples not found in {SAMPLES_DIR}. Run extraction script in notebook first.")

    with open(records_path, "rb") as f:
        records = pickle.load(f)
    with open(syn_params_path, "rb") as f:
        syn_params = pickle.load(f)
    with open(labels_path, "rb") as f:
        labels = pickle.load(f)

    return records, syn_params, labels


def test_detection_method_consistency_real_data(real_data):
    """
    Verifies that 'Iterative' (Option A) and 'Merged' (Option B) detection methods
    produce identical results using a subset of real simulation data.
    """
    records, syn_params, labels = real_data

    layer = 4
    nrn_id = 3012
    w_min = 0.5
    tol = 3.0
    offset = 50.0
    duration = records.item(0).duration - offset

    # Initialize Database
    rdb = ResultsDatabase(
        records, syn_params, labels, layer=slice(None),
        proj=('FF', 'E2E'), duration=duration, offset=offset
    )

    # Define Refinement Pipeline
    index = (layer, nrn_id)
    refine_steps = [
        refinery.DropRepeating(),
        refinery.FilterLayers([layer - 1, layer, layer]),
        refinery.FilterIndex(index, position=1),
        refinery.Constrained(syn_params, w_min, tol),
        refinery.Merge(tol=tol, strategy='mean'),
        refinery.Constrained(syn_params, w_min, tol)
    ]
    targets = {}

    # --- Option A: Iterative Detection ---
    polygrps_A = detection.mine_triads(nrn_id, layer, rdb, targets, w_min, tol)

    # --- Option B: Merged Detection ---
    polygrps_B = detection.mine_structural(nrn_id, layer, rdb, targets, w_min, tol=tol)

    # --- Assertions ---
    # 1. Check counts
    assert len(polygrps_A) == len(polygrps_B), (
        f"Mismatch in number of PNGs found: Iterative={len(polygrps_A)}, Merged={len(polygrps_B)}"
    )

    # 2. Check content equality (using set comparison)
    assert set(polygrps_A) == set(polygrps_B), " The set of detected PNGs differs between methods."
