#!/bin/bash
# Market Wizard - Skrypt uruchomieniowy

set -e

# Kolory
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔮 Market Wizard${NC}"
echo "================================"

# Katalog skryptu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Sprawdź czy venv istnieje
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Tworzę środowisko wirtualne...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip -q
    echo -e "${GREEN}📦 Instaluję zależności (może potrwać kilka minut)...${NC}"
    cd backend && pip install -e . && cd ..
else
    source venv/bin/activate
fi

# Sprawdź czy .env istnieje
if [ ! -f "backend/.env" ]; then
    echo -e "${YELLOW}📋 Tworzę plik .env z szablonu...${NC}"
    cp backend/.env.example backend/.env
    echo -e "${YELLOW}⚠️  Uzupełnij GOOGLE_API_KEY w backend/.env${NC}"
    echo ""
    echo "Otwórz backend/.env i dodaj swój klucz API Google:"
    echo "  GOOGLE_API_KEY=your-api-key-here"
    echo ""
    exit 1
fi

# Wybór trybu
MODE=${1:-gradio}

if [ "$MODE" = "api" ]; then
    echo -e "${GREEN}🚀 Uruchamiam API (FastAPI)...${NC}"
    echo "Dokumentacja: http://localhost:8000/docs"
    cd backend
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
elif [ "$MODE" = "gradio" ]; then
    echo -e "${GREEN}🚀 Uruchamiam UI (Gradio)...${NC}"
    echo "Otwórz: http://localhost:7860"
    cd frontend
    python main.py
elif [ "$MODE" = "test" ]; then
    echo -e "${GREEN}🧪 Uruchamiam testy...${NC}"
    cd backend
    python -m pytest tests/ -v
else
    echo "Użycie: ./run.sh [api|gradio|test]"
    echo ""
    echo "  api    - Uruchom API FastAPI (port 8000)"
    echo "  gradio - Uruchom interfejs Gradio (port 7860)"
    echo "  test   - Uruchom testy"
    exit 1
fi
