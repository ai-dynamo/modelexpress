# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/bin/bash
# Bash completion script for model-express-cli
# Source this file or add it to your .bashrc:
#   source /path/to/model-express-cli-completion.bash

_model_express_cli_completions() {
    local cur prev words cword
    _init_completion || return

    # Main commands
    local commands="health model api help"
    local global_opts="--endpoint --timeout --format --verbose --quiet --cache-path --help --version"
    local formats="human json json-pretty"

    case "${COMP_CWORD}" in
        1)
            COMPREPLY=($(compgen -W "$commands" -- "$cur"))
            return 0
            ;;
        *)
            case "${prev}" in
                --endpoint|-e)
                    COMPREPLY=($(compgen -W "http://localhost:8001 https://" -- "$cur"))
                    return 0
                    ;;
                --timeout|-t)
                    COMPREPLY=($(compgen -W "30 60 120" -- "$cur"))
                    return 0
                    ;;
                --format|-f)
                    COMPREPLY=($(compgen -W "$formats" -- "$cur"))
                    return 0
                    ;;
                --payload|-p)
                    COMPREPLY=($(compgen -W '"{}"' -- "$cur"))
                    return 0
                    ;;
                --payload-file)
                    COMPREPLY=($(compgen -f -- "$cur"))
                    return 0
                    ;;
                --cache-path)
                    COMPREPLY=($(compgen -d -- "$cur"))
                    return 0
                    ;;
            esac
            ;;
    esac

    # Handle subcommands
    local command=""
    local i
    for (( i=1; i < COMP_CWORD; i++ )); do
        if [[ "${words[i]}" =~ ^(health|model|api)$ ]]; then
            command="${words[i]}"
            break
        fi
    done

    case "$command" in
        health)
            if [[ "$cur" == -* ]]; then
                COMPREPLY=($(compgen -W "--help" -- "$cur"))
            fi
            ;;
        model)
            local model_commands="download init list status clear clearall validate stats help"
            if [[ "${words[i+1]}" == "" || "${words[i+1]}" == "$cur" ]]; then
                COMPREPLY=($(compgen -W "$model_commands" -- "$cur"))
            elif [[ "${words[i+1]}" == "download" ]]; then
                case "${prev}" in
                    --provider|-p)
                        COMPREPLY=($(compgen -W "hugging-face" -- "$cur"))
                        ;;
                    --strategy|-s)
                        COMPREPLY=($(compgen -W "smart-fallback server-fallback server-only direct" -- "$cur"))
                        ;;
                    download)
                        # Common model names for completion
                        COMPREPLY=($(compgen -W "google-t5/t5-small microsoft/DialoGPT-small microsoft/DialoGPT-medium" -- "$cur"))
                        ;;
                    *)
                        if [[ "$cur" == -* ]]; then
                            COMPREPLY=($(compgen -W "--provider --strategy --help" -- "$cur"))
                        fi
                        ;;
                esac
            elif [[ "${words[i+1]}" == "init" ]]; then
                case "${prev}" in
                    --storage-path)
                        COMPREPLY=($(compgen -d -- "$cur"))
                        ;;
                    --server-endpoint)
                        COMPREPLY=($(compgen -W "http://localhost:8001 https://" -- "$cur"))
                        ;;
                    *)
                        if [[ "$cur" == -* ]]; then
                            COMPREPLY=($(compgen -W "--storage-path --server-endpoint --auto-mount --help" -- "$cur"))
                        fi
                        ;;
                esac
            elif [[ "${words[i+1]}" == "list" ]]; then
                if [[ "$cur" == -* ]]; then
                    COMPREPLY=($(compgen -W "--detailed --help" -- "$cur"))
                fi
            elif [[ "${words[i+1]}" == "clear" ]]; then
                case "${prev}" in
                    clear)
                        # Could potentially list actual downloaded models here
                        COMPREPLY=($(compgen -W "google-t5/t5-small microsoft/DialoGPT-small" -- "$cur"))
                        ;;
                    *)
                        if [[ "$cur" == -* ]]; then
                            COMPREPLY=($(compgen -W "--help" -- "$cur"))
                        fi
                        ;;
                esac
            elif [[ "${words[i+1]}" == "clearall" ]]; then
                if [[ "$cur" == -* ]]; then
                    COMPREPLY=($(compgen -W "--yes --help" -- "$cur"))
                fi
            elif [[ "${words[i+1]}" == "validate" ]]; then
                case "${prev}" in
                    validate)
                        # Could potentially list actual downloaded models here
                        COMPREPLY=($(compgen -W "google-t5/t5-small microsoft/DialoGPT-small" -- "$cur"))
                        ;;
                    *)
                        if [[ "$cur" == -* ]]; then
                            COMPREPLY=($(compgen -W "--help" -- "$cur"))
                        fi
                        ;;
                esac
            elif [[ "${words[i+1]}" == "stats" ]]; then
                if [[ "$cur" == -* ]]; then
                    COMPREPLY=($(compgen -W "--detailed --help" -- "$cur"))
                fi
            elif [[ "${words[i+1]}" == "status" ]]; then
                if [[ "$cur" == -* ]]; then
                    COMPREPLY=($(compgen -W "--help" -- "$cur"))
                fi
            fi
            ;;
        api)
            local api_commands="send help"
            if [[ "${words[i+1]}" == "" || "${words[i+1]}" == "$cur" ]]; then
                COMPREPLY=($(compgen -W "$api_commands" -- "$cur"))
            elif [[ "${words[i+1]}" == "send" ]]; then
                case "${prev}" in
                    send)
                        COMPREPLY=($(compgen -W "ping" -- "$cur"))
                        ;;
                    *)
                        if [[ "$cur" == -* ]]; then
                            COMPREPLY=($(compgen -W "--payload --payload-file --help" -- "$cur"))
                        fi
                        ;;
                esac
            fi
            ;;
    esac

    # Global options
    if [[ "$cur" == -* ]]; then
        COMPREPLY+=($(compgen -W "$global_opts" -- "$cur"))
    fi
}

complete -F _model_express_cli_completions model-express-cli
