echo "Setting up ClipView..."
echo "========================="

echo "✅ Python found: $(python3 --version)"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

if [ $? -ne 0 ]; then
    echo "❌ Failed to install PyTorch. Trying CPU version..."
    pip install torch torchvision
fi

pip install -r segmentation_scripts/requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements. Please check requirements.txt exists."
    exit 1
fi

# Install FFmpeg
echo ""
echo "Installing FFmpeg..."

# Install FFmpeg using apt-get (Ubuntu/Debian)
if command -v apt-get &> /dev/null; then
    echo "Detected Ubuntu/Debian - installing via apt"
    sudo apt update
    sudo apt install -y ffmpeg
else
    echo "⚠️  This script is designed for Ubuntu/Debian systems with apt-get."
    echo "Please install FFmpeg manually for your system:"
    echo "   For more info: https://ffmpeg.org/download.html"
fi

# Verify FFmpeg installation
echo ""
echo "🔍 Verifying FFmpeg installation..."

if command -v ffmpeg &> /dev/null; then
    echo "✅ FFmpeg found: $(which ffmpeg)"
    echo "   Version: $(ffmpeg -version | head -n1)"
else
    echo "❌ FFmpeg not found. Please install it manually."
    echo "   For more info: https://ffmpeg.org/download.html"
fi