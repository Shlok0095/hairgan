@echo off
echo Fixing Flask compatibility issues...

pip uninstall -y flask flask-restx Werkzeug
pip install flask==2.0.1 flask-restx==0.5.1 Werkzeug==2.0.1

echo Done! Now try running the app with: python app.py
pause 