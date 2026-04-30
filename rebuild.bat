@echo off
echo Rebuilding model data...
python scripts/build_model_data.py
if %errorlevel% neq 0 (
    echo Data build failed.
    pause
    exit /b 1
)

echo Starting dev server...
npm run dev
