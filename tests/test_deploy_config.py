from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_render_blueprint_is_a_free_singapore_web_service():
    text = (ROOT / "render.yaml").read_text()
    assert "type: web" in text
    assert "runtime: python" in text
    assert "plan: free" in text
    assert "region: singapore" in text
    assert "healthCheckPath: /health" in text
    assert "0.0.0.0" in text
    assert "$PORT" in text
    assert "key: GROQ_API_KEY" in text
    assert "sync: false" in text
    assert "gsk_" not in text
    assert "sk-" not in text


def test_dockerfile_binds_render_port():
    text = (ROOT / "Dockerfile").read_text()
    assert "0.0.0.0" in text
    assert "${PORT:-10000}" in text
    assert "torch" not in text.lower()


def test_env_example_has_groq_placeholder_only():
    text = (ROOT / ".env.example").read_text()
    assert "GROQ_API_KEY=" in text
    assert "gsk_" not in text
    assert "sk-" not in text
    assert ".env" in (ROOT / ".gitignore").read_text()


def test_runtime_deps_stay_small():
    requirements = (ROOT / "requirements.txt").read_text()
    indexing = (ROOT / "src" / "indexing.py").read_text()
    assert "torch" not in requirements
    assert "faiss" not in requirements
    assert "fastembed" not in requirements
    assert "import torch" not in indexing
    assert "sentence_transformers" not in indexing


def test_no_huggingface_space_sync_workflow():
    workflows = list((ROOT / ".github" / "workflows").glob("*.yml"))
    combined = "\n".join(path.read_text() for path in workflows)
    assert "huggingface.co/spaces" not in combined
    assert "HF_SPACE" not in combined
