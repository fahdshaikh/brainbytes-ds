#!/bin/bash
# 🚀 One-Command Setup Script for Data Science MiniCourse
# ======================================================
# This script sets up everything you need to start learning!

set -euo pipefail

# 🎨 Colors for beautiful output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                                                              ║"
    echo "║          🎓 Data Science MiniCourse Setup                     ║"
    echo "║                                                              ║"
    echo "║          Ready to transform into a data wizard? 🧙‍♂️          ║"
    echo "║                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "\n${BLUE}🔧 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 🎯 Main setup function
main() {
    print_header
    
    print_step "Checking Python installation..."
    
    # Check if Python 3.8+ is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found!"
        echo -e "${YELLOW}Please install Python 3.8+ first:${NC}"
        echo "  • macOS: brew install python3"
        echo "  • Ubuntu: sudo apt install python3 python3-pip"
        echo "  • Windows: Download from python.org"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Found Python $PYTHON_VERSION"
    
    # Check Python version compatibility
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
        print_warning "Python 3.8+ recommended for best compatibility"
    fi
    
    print_step "Setting up virtual environment..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d ".venv" ]]; then
        python3 -m venv .venv
        print_success "Created virtual environment"
    else
        print_success "Virtual environment already exists"
    fi
    
    print_step "Activating environment and installing packages..."
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Upgrade pip first
    python -m pip install --upgrade pip --quiet
    
    # Install core dependencies
    echo -e "${CYAN}📦 Installing core learning packages...${NC}"
    pip install -r requirements.txt --quiet
    print_success "Core packages installed"
    
    # Ask if they want dev tools
    echo -e "\n${YELLOW}🛠️  Do you want development tools? (y/N)${NC}"
    echo "   (Jupyter, testing tools, code formatters)"
    read -r -n 1 install_dev
    echo
    
    if [[ $install_dev =~ ^[Yy]$ ]]; then
        print_step "Installing development tools..."
        pip install -r requirements-dev.txt --quiet
        print_success "Development tools installed"
    fi
    
    print_step "Creating helpful shortcuts..."
    
    # Create activation script
    cat > activate_env.sh << 'EOF'
#!/bin/bash
# 🚀 Quick activation script
echo "🧪 Activating Data Science environment..."
source .venv/bin/activate
echo "✅ Environment active!"
echo "🎯 Quick commands:"
echo "   python 1_python_fundamentals/1_python_fundamentals.py"
echo "   jupyter lab"
echo "   jupyter notebook"
EOF
    chmod +x activate_env.sh
    
    print_step "Testing installation..."
    
    # Quick test
    if python -c "import numpy, pandas, matplotlib, seaborn, sklearn, scipy; print('✅ All packages imported successfully!')" 2>/dev/null; then
        print_success "Installation test passed!"
    else
        print_error "Some packages failed to import"
        echo -e "${YELLOW}Try running: pip install -r requirements.txt${NC}"
        exit 1
    fi
    
    # Final success message
    echo -e "\n${GREEN}🎉 SETUP COMPLETE! Your learning environment is ready!${NC}"
    echo -e "\n${CYAN}🚀 Quick Start:${NC}"
    echo "  1. Activate environment: source .venv/bin/activate"
    echo "  2. Or use shortcut: ./activate_env.sh"
    echo "  3. Start learning: cd 1_python_fundamentals"
    echo "  4. Run lesson: python 1_python_fundamentals.py"
    echo "  5. Or use Jupyter: jupyter lab"
    echo -e "\n${PURPLE}📚 Learning Path:${NC}"
    echo "  🐍 1_python_fundamentals → 📊 2_data_manipulation → 📈 3_data_visualization → 🤖 4_statistics_ml"
    echo -e "\n${GREEN}Happy learning! 🎓✨${NC}"
}

# 🔒 Safety check - make sure we're in the right directory
if [[ ! -f "requirements.txt" ]]; then
    print_error "Not in the right directory!"
    echo "Please run this script from the brainbytes-ds root folder"
    exit 1
fi

# 🎬 Run the main setup
main "$@"
