#!/bin/bash
# Package trained models for sharing with team

echo "📦 Packaging trained models..."

# Check if models exist
if [ ! -d "runs/movielens" ]; then
    echo "❌ Error: No models found in runs/movielens/"
    echo "   Train models first with: make train MODEL=... DATA=movielens_1m"
    exit 1
fi

# Create output directory
mkdir -p shared_models

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="shared_models/trained_models_1m_${TIMESTAMP}.zip"

# Package all models
echo "Zipping models from runs/movielens/..."
cd runs && zip -r "../${OUTPUT_FILE}" movielens/ && cd ..

# Get file size
SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)

echo ""
echo "✅ Success! Models packaged to: ${OUTPUT_FILE}"
echo "📊 File size: ${SIZE}"
echo ""
echo "📤 Next steps:"
echo "1. Upload ${OUTPUT_FILE} to Google Drive"
echo "2. Get shareable link (Anyone with link can view)"
echo "3. Share with your team!"
echo ""
echo "🔗 Or use this command to upload via gdrive CLI (if installed):"
echo "   gdrive upload ${OUTPUT_FILE}"
