"""Checkpoint save/load roundtrip: architecture metadata must survive."""
import torch
import pytest

from src.models.multitask import EyeBagModel, load_model_from_checkpoint


def _tiny_model(**overrides):
    kwargs = dict(
        encoder_name="resnet18",
        pretrained=False,
        num_grades=5,
        proj_dim=64,          # non-default on purpose
        dropout=0.1,
        use_severity=True,
        use_dark_circles=False,
    )
    kwargs.update(overrides)
    return EyeBagModel(**kwargs)


def _save_checkpoint(model, path, model_config):
    torch.save(
        {"model_state": model.state_dict(), "epoch": 3, "model_config": model_config},
        path,
    )


def test_roundtrip_with_model_config(tmp_path):
    """Non-default proj_dim/heads must reload exactly via model_config."""
    model = _tiny_model()
    ckpt_path = tmp_path / "best.pt"
    _save_checkpoint(model, ckpt_path, {
        "encoder": "resnet18",
        "pretrained": False,
        "severity_grades": 5,
        "proj_dim": 64,
        "dropout": 0.1,
        "use_severity_head": True,
        "use_dark_circles_head": False,
    })

    loaded = load_model_from_checkpoint(str(ckpt_path), torch.device("cpu"))
    assert loaded.encoder_name == "resnet18"
    assert loaded.severity_head is not None
    assert loaded.dark_circles_head is None

    x = torch.randn(2, 3, 160, 256)
    with torch.no_grad():
        a = model.eval()(x)
        b = loaded(x)
    assert torch.allclose(a["presence_logit"], b["presence_logit"], atol=1e-6)
    assert torch.allclose(a["severity_logits"], b["severity_logits"], atol=1e-6)


def test_legacy_checkpoint_fallback(tmp_path):
    """Checkpoints without model_config still load via state-dict sniffing
    (only for default proj_dim and resnet18/convnext_tiny encoders)."""
    model = _tiny_model(proj_dim=512, dropout=0.2)
    ckpt_path = tmp_path / "legacy.pt"
    torch.save({"model_state": model.state_dict(), "epoch": 1}, ckpt_path)

    loaded = load_model_from_checkpoint(str(ckpt_path), torch.device("cpu"))
    assert loaded.encoder_name == "resnet18"
    assert loaded.severity_head is not None
    assert loaded.dark_circles_head is None


def test_nondefault_arch_without_config_fails_loudly(tmp_path):
    """A non-default proj_dim with no model_config must raise, not silently
    load a mismatched architecture."""
    model = _tiny_model()  # proj_dim=64
    ckpt_path = tmp_path / "no_meta.pt"
    torch.save({"model_state": model.state_dict()}, ckpt_path)

    with pytest.raises(RuntimeError):
        load_model_from_checkpoint(str(ckpt_path), torch.device("cpu"))


def test_return_checkpoint_flag(tmp_path):
    model = _tiny_model()
    ckpt_path = tmp_path / "best.pt"
    _save_checkpoint(model, ckpt_path, {
        "encoder": "resnet18", "pretrained": False, "severity_grades": 5,
        "proj_dim": 64, "dropout": 0.1,
        "use_severity_head": True, "use_dark_circles_head": False,
    })
    loaded, ckpt = load_model_from_checkpoint(
        str(ckpt_path), torch.device("cpu"), return_checkpoint=True
    )
    assert ckpt["epoch"] == 3
