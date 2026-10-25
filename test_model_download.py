"""
Checks the Hugging Face model download logic without any network access.

    python test_model_download.py
"""

import shutil
import tempfile
from pathlib import Path
from unittest import mock

import huggingface_hub
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

import main

REAL_MODEL_DIR = main.MODEL_DIR
SECRET = "hf_SECRET_VALUE_FOR_TEST"


def run(env, fake_download, tmp):
    main.MODEL_DIR = Path(tmp) / "model"
    with mock.patch.dict("os.environ", env, clear=False), mock.patch.object(
        huggingface_hub, "hf_hub_download", side_effect=fake_download
    ) as dl:
        try:
            main.ensure_model_files()
            return None, dl
        except RuntimeError as exc:
            return str(exc), dl


def copy_from_local(repo_id, filename, token, local_dir):
    shutil.copy(REAL_MODEL_DIR / filename, Path(local_dir) / filename)


def make_error(cls):
    """Build a huggingface_hub error across library versions (constructor differs)."""
    try:
        return cls("x", response=FakeResponse())
    except TypeError:
        return cls("x")


def clear_env():
    return {"HF_MODEL_REPO": "", "HF_TOKEN": ""}


class FakeResponse:
    status_code = 401
    headers = {}
    request = None
    text = ""

    def json(self):
        return {}


with tempfile.TemporaryDirectory() as tmp:
    # 1. Files already present: nothing is downloaded (local development).
    main.MODEL_DIR = REAL_MODEL_DIR
    with mock.patch.object(huggingface_hub, "hf_hub_download") as dl:
        main.ensure_model_files()
        assert dl.call_count == 0
    print("local files present -> no download OK")

    # 2. Files missing + repo configured: all three are downloaded, token passed securely.
    err, dl = run(
        {"HF_MODEL_REPO": "user/repo", "HF_TOKEN": SECRET}, copy_from_local, tmp + "/a"
    )
    assert err is None and dl.call_count == 3
    assert all(
        c.kwargs["token"] == SECRET and c.kwargs["repo_id"] == "user/repo"
        for c in dl.call_args_list
    )
    assert all((Path(tmp, "a", "model", n)).is_file() for n in main.MODEL_FILES)
    print("download of 3 files OK")

    # 3. Missing HF_MODEL_REPO gives a clear message.
    err, _ = run(clear_env(), copy_from_local, tmp + "/b")
    assert "HF_MODEL_REPO is not set" in err
    print("missing HF_MODEL_REPO message OK")

    # 4. Private repo without token / wrong repo or token.
    err, _ = run(
        {"HF_MODEL_REPO": "user/repo", "HF_TOKEN": ""},
        make_error(RepositoryNotFoundError),
        tmp + "/c",
    )
    assert "HF_TOKEN is not set" in err
    err, _ = run(
        {"HF_MODEL_REPO": "user/repo", "HF_TOKEN": SECRET},
        make_error(RepositoryNotFoundError),
        tmp + "/d",
    )
    assert "not found or is not accessible" in err and SECRET not in err
    print("repo not found / no token messages OK")

    # 5. File missing in repo, auth error, network failure.
    err, _ = run(
        {"HF_MODEL_REPO": "user/repo", "HF_TOKEN": SECRET},
        make_error(EntryNotFoundError),
        tmp + "/e",
    )
    assert "does not exist in Hugging Face repo" in err
    http_err = huggingface_hub.errors.HfHubHTTPError("401", response=FakeResponse())
    err, _ = run(
        {"HF_MODEL_REPO": "user/repo", "HF_TOKEN": SECRET}, http_err, tmp + "/f"
    )
    assert "authentication failed" in err and SECRET not in err
    err, _ = run(
        {"HF_MODEL_REPO": "user/repo", "HF_TOKEN": SECRET},
        ConnectionError("down"),
        tmp + "/g",
    )
    assert "ConnectionError" in err and SECRET not in err
    print("missing file / auth / network messages OK")

main.MODEL_DIR = REAL_MODEL_DIR
print("\nAll model download tests passed!")
