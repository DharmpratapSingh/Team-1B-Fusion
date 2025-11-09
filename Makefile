serve:
	uv run uvicorn mcp_server:app --host 127.0.0.1 --port $${PORT:-8010} --reload

ui:
	uv run streamlit run enhanced_climategpt_with_personas.py

test:
	uv run pytest -q
