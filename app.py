"""Entry point:  uvicorn app:app --reload   ->  http://localhost:8000"""
from voicedesk.server import create_app

app = create_app()
