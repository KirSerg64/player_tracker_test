#!/bin/bash

# Setup script for Video Processing Pipeline on Linux

echo "=== Video Processing Pipeline Setup for Linux ==="

# Check if running as root (not recommended)
if [[ $EUID -eq 0 ]]; then
    echo "⚠️  Warning: Running as root is not recommended"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Function to check command availability
check_command() {
    if command -v "$1" &> /dev/null; then
        echo "✅ $1 is installed"
        return 0
    else
        echo "❌ $1 is not installed"
        return 1
    fi
}

# Function to install packages based on distribution
install_package() {
    local package=$1
    
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        echo "Installing $package using apt-get..."
        sudo apt-get update && sudo apt-get install -y "$package"
    elif command -v yum &> /dev/null; then
        # RHEL/CentOS
        echo "Installing $package using yum..."
        sudo yum install -y "$package"
    elif command -v dnf &> /dev/null; then
        # Fedora
        echo "Installing $package using dnf..."
        sudo dnf install -y "$package"
    elif command -v pacman &> /dev/null; then
        # Arch Linux
        echo "Installing $package using pacman..."
        sudo pacman -S --noconfirm "$package"
    else
        echo "❌ Cannot determine package manager. Please install $package manually."
        return 1
    fi
}

echo "🔍 Checking dependencies..."

# Check Python
if ! check_command python3; then
    echo "Installing Python 3..."
    install_package python3
fi

# Check pip
if ! check_command pip3; then
    echo "Installing pip..."
    if command -v apt-get &> /dev/null; then
        install_package python3-pip
    else
        install_package python3-pip
    fi
fi

# Check FFmpeg
if ! check_command ffmpeg; then
    echo "Installing FFmpeg..."
    install_package ffmpeg
fi

# Check git (might be needed for dependencies)
if ! check_command git; then
    echo "Installing git..."
    install_package git
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install --user pyyaml pandas matplotlib seaborn numpy pickle-mixin loguru tqdm pillow

# Make the pipeline script executable
if [[ -f "run_video_pipeline.sh" ]]; then
    chmod +x run_video_pipeline.sh
    echo "✅ Made run_video_pipeline.sh executable"
else
    echo "❌ run_video_pipeline.sh not found in current directory"
fi

# Create example usage script
cat > run_example.sh << 'EOF'
#!/bin/bash

# Example usage of the video processing pipeline

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Example video path - modify this to your actual video file
VIDEO_PATH="/path/to/your/football_match.mp4"

# Check if video exists
if [[ ! -f "$VIDEO_PATH" ]]; then
    echo "❌ Please modify VIDEO_PATH in this script to point to your video file"
    echo "Current path: $VIDEO_PATH"
    exit 1
fi

# Run the pipeline with default settings
echo "🚀 Running video processing pipeline..."
./run_video_pipeline.sh --video "$VIDEO_PATH"

# Alternative: Run with custom settings
# ./run_video_pipeline.sh --video "$VIDEO_PATH" --output "my_output" --segment-duration 300 --max-parallel 6

echo "✅ Pipeline completed! Check the output directory for results."
EOF

chmod +x run_example.sh
echo "✅ Created run_example.sh with usage example"

# Check system resources
echo "🖥️  System Information:"
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "Disk space: $(df -h . | tail -1 | awk '{print $4}') available"

# Performance recommendations
echo ""
echo "🔧 Performance Recommendations:"
CORES=$(nproc)
if [[ $CORES -gt 4 ]]; then
    RECOMMENDED_PARALLEL=$((CORES / 2))
    echo "• Your system has $CORES cores. Consider using --max-parallel $RECOMMENDED_PARALLEL"
else
    echo "• Your system has $CORES cores. The default --max-parallel 4 should work fine"
fi

MEMORY_GB=$(free -g | awk '/^Mem:/ {print $2}')
if [[ $MEMORY_GB -lt 8 ]]; then
    echo "• You have ${MEMORY_GB}GB RAM. Consider using shorter segment durations (--segment-duration 300)"
else
    echo "• You have ${MEMORY_GB}GB RAM. Default settings should work well"
fi

echo ""
echo "🎯 Quick Start:"
echo "1. Edit run_example.sh to set your video path"
echo "2. Run: ./run_example.sh"
echo "3. Or run directly: ./run_video_pipeline.sh --video /path/to/video.mp4"

echo ""
echo "📁 Make sure your project structure includes:"
echo "   ├── sn_gamestate/"
echo "   ├── pretrained_models/"
echo "   ├── main.py"
echo "   └── run_video_pipeline.sh"

echo ""
echo "✅ Setup complete! Ready to process videos."
