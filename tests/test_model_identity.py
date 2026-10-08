import torch
from safetensors.torch import save_file

from scripts.check_weight_change import audit, compare_weights
from scripts.model_identity import model_identity


def test_real_tensor_weight_delta_and_no_update_control(tmp_path):
    base = tmp_path / "base"
    trained = tmp_path / "trained"
    control = tmp_path / "control"
    for root in (base, trained, control):
        root.mkdir()
    a = {"model.weight": torch.tensor([1.0, 2.0, 3.0])}
    b = {"model.weight": torch.tensor([1.0, 2.5, 3.0])}
    save_file(a, str(base / "model.safetensors"))
    save_file(b, str(trained / "model.safetensors"))
    save_file(a, str(control / "model.safetensors"))
    change = compare_weights(base, trained)
    assert change["changed_elements"] == 1
    assert audit(base, trained, control)["passed"]
    identity = model_identity(base, original="local-test", revision="LOCAL_WEIGHT_HASH")
    assert identity["weight_file_count"] == 1
    assert len(identity["weights_fingerprint_sha256"]) == 64


def test_no_update_changed_is_rejected(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    save_file({"w": torch.tensor([1.0])}, str(a / "model.safetensors"))
    save_file({"w": torch.tensor([9.0])}, str(b / "model.safetensors"))
    assert not audit(a, b, b)["passed"]
