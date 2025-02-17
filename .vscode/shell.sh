#!/bin/bash

# Detect shell
USER_SHELL=$(basename "$SHELL")

case "$USER_SHELL" in
    "bash")
        if [[ -f ~/.bashrc ]]; then
            source ~/.bashrc
        fi
        ;;
    "zsh")
        if [[ -f ~/.zshrc ]]; then
            source ~/.zshrc
        fi
        ;;
esac

# Detect OS
OS_TYPE=$(uname -s)
HOSTNAME=$(hostname)

case "$OS_TYPE" in
    "Linux")
        case "$HOSTNAME" in
            "deepmeme")
                echo "[Linux] Setting up for deepmeme..."
                eval "$(micromamba shell hook --shell bash)"
                micromamba activate ai_env
                ;;
            *)
                echo "[Linux] Unknown host: $HOSTNAME. Using default shell. (see .vscode/settings.json and .vscode/shell.sh for more details)"
                ;;
        esac
        ;;
    
    "Darwin")
        case "$HOSTNAME" in
            "wwmbp")
                echo "[macOS] Setting up for wwmbp..."
                nix-shell
                ;;
            *)
                echo "[macOS] Unknown host: $HOSTNAME. Using default shell. (see .vscode/settings.json and .vscode/shell.sh for more details)"
                ;;
        esac
        ;;
    
    *)
        echo "[Unknown OS] $OS_TYPE detected. No custom setup. (see .vscode/settings.json and .vscode/shell.sh for more details)"
        ;;
esac