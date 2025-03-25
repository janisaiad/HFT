pip install uv


export UV_LINK_MODE=symlink
source ~/.zshrc
echo $UV_LINK_MODE


uv venv
source .venv/bin/activate


uv --link-mode=copysync
uv pip install -e .
uv cache prune



uv run tests/test_env.py
source .venv/bin/activate
