"""
WSGI Entry point for production servers (Gunicorn, uWSGI, Render, Railway).
"""
from app import app

if __name__ == "__main__":
    app.run()
