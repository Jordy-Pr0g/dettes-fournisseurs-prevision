@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Premiere installation, cela peut prendre quelques minutes...
  python -m venv .venv
  if errorlevel 1 goto erreur
  ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
  if errorlevel 1 goto erreur
)
echo L'application va s'ouvrir dans le navigateur. Ferme cette fenetre pour l'arreter.
start "" /b cmd /c "timeout /t 8 >nul & start http://localhost:8501"
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless true --server.address localhost
goto fin
:erreur
echo L'installation a echoue. Verifie que Python est installe et que tu es connecte a internet.
pause
:fin
