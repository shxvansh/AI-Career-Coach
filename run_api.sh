#!/bin/bash
# Install new dependency if missing
pip install python-multipart

# Run the API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
