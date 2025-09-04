#!/bin/bash
# 🚀 Post-Creation Setup Script for Data Science Learning Environment

echo "🎯 Setting up your Data Science Learning Lab..."

# 📦 Install core dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 🧪 Install development tools (optional)
echo "🧪 Installing development tools..."
pip install -r requirements-dev.txt

# 🎨 Set up Git (if not already configured)
echo "🎨 Configuring Git..."
git config --global --get user.name || git config --global user.name "Data Science Learner"
git config --global --get user.email || git config --global user.email "learner@example.com"

# 📚 Create helpful aliases
echo "📚 Setting up helpful commands..."
cat >> ~/.bashrc << 'EOF'

# 🎯 Data Science Learning Aliases
alias lab='jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
alias notebook='jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
alias run-all='for dir in */; do echo "🚀 Running $dir"; cd "$dir" && python *.py && cd ..; done'
alias install-dev='pip install -r requirements-dev.txt'
alias test-code='flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics'

# 🎨 Make terminal colorful and friendly
PS1='\[\e[1;32m\]🧪 \[\e[1;34m\]\w\[\e[1;32m\] $ \[\e[0m\]'

EOF

# 📝 Create welcome message
cat > WELCOME_TO_CODESPACES.md << 'EOF'
# 🎉 Welcome to Your Data Science Learning Lab!

Your environment is ready! Here's what you can do:

## 🚀 Quick Start Commands:
```bash
# Launch Jupyter Lab
lab

# Run all Python scripts to test
run-all

# Check code quality
test-code
```

## 📚 Learning Path:
1. Start with `1_python_fundamentals/`
2. Move to `2_data_manipulation/` 
3. Continue to `3_data_visualization/`
4. Finish with `4_statistics_ml/`

## 💡 Pro Tips:
- Use Ctrl+Shift+` to open new terminal
- Install additional packages with `pip install package-name`
- Save your work - Codespaces persists for 30 days of inactivity

Happy Learning! 🎓✨
EOF

echo "✅ Setup complete! Your Data Science Learning Lab is ready!"
echo "📚 Check out WELCOME_TO_CODESPACES.md for quick start tips!"
