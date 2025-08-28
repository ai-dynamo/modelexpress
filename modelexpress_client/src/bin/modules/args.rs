// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::{Parser, Subcommand, ValueEnum};
use modelexpress_client::ModelProvider;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "model-express-cli")]
#[command(about = "A CLI tool for interacting with ModelExpress server")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// Server endpoint (can also be set via MODEL_EXPRESS_ENDPOINT env var)
    #[arg(
        long,
        short = 'e',
        env = "MODEL_EXPRESS_ENDPOINT",
        default_value = "http://localhost:8001"
    )]
    pub endpoint: String,

    /// Request timeout in seconds
    #[arg(long, short = 't', default_value = "30")]
    pub timeout: u64,

    /// Output format
    #[arg(long, short = 'f', value_enum, default_value = "human")]
    pub format: OutputFormat,

    /// Verbose mode (-v for info, -vv for debug, -vvv for trace)
    #[arg(short = 'v', action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Quiet mode (suppress all output except errors)
    #[arg(long, short = 'q')]
    pub quiet: bool,

    /// Cache path override
    #[arg(long, value_name = "PATH")]
    pub cache_path: Option<PathBuf>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Check server health and status
    Health,

    /// Model management operations (download, list, clear, validate, etc.)
    Model {
        #[command(subcommand)]
        command: ModelCommands,
    },

    /// Send general API requests
    Api {
        #[command(subcommand)]
        command: ApiCommands,
    },
}

#[derive(Subcommand)]
pub enum ModelCommands {
    /// Download a model with various strategies (automatically cached)
    Download {
        /// Name of the model to download
        model_name: String,

        /// Model provider
        #[arg(long, short = 'p', value_enum, default_value = "hugging-face")]
        provider: CliModelProvider,

        /// Download strategy
        #[arg(long, short = 's', value_enum, default_value = "smart-fallback")]
        strategy: DownloadStrategy,
    },

    /// Initialize model storage configuration
    Init {
        /// Storage path for models
        #[arg(long, value_name = "PATH")]
        storage_path: Option<PathBuf>,

        /// Server endpoint
        #[arg(long, value_name = "ENDPOINT")]
        server_endpoint: Option<String>,
    },

    /// List downloaded models
    List {
        /// Show detailed information
        #[arg(long)]
        detailed: bool,
    },

    /// Show model storage status and usage
    Status,

    /// Clear specific model from storage
    Clear {
        /// Model name to clear
        model_name: String,
    },

    /// Clear all models from storage
    ClearAll {
        /// Confirm without prompting
        #[arg(long)]
        yes: bool,
    },

    /// Validate model integrity
    Validate {
        /// Model name to validate (optional)
        model_name: Option<String>,
    },

    /// Show model storage statistics
    Stats {
        /// Show detailed statistics
        #[arg(long)]
        detailed: bool,
    },
}

#[derive(Subcommand)]
pub enum ApiCommands {
    /// Send a custom API request
    Send {
        /// The action to perform
        action: String,

        /// JSON payload (use - to read from stdin)
        #[arg(long, short = 'p')]
        payload: Option<String>,

        /// Read payload from file
        #[arg(long, short = 'f')]
        payload_file: Option<String>,
    },
}

#[derive(ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    /// Human-readable output with colors
    Human,
    /// JSON output
    Json,
    /// Pretty-printed JSON
    JsonPretty,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum CliModelProvider {
    #[value(name = "hugging-face")]
    HuggingFace,
}

impl From<CliModelProvider> for ModelProvider {
    fn from(provider: CliModelProvider) -> Self {
        match provider {
            CliModelProvider::HuggingFace => ModelProvider::HuggingFace,
        }
    }
}

#[derive(ValueEnum, Clone, Debug)]
pub enum DownloadStrategy {
    /// Try server first, fallback to direct download if needed
    #[value(name = "smart-fallback")]
    SmartFallback,
    /// Use server only (no fallback)
    #[value(name = "server-only")]
    ServerOnly,
    /// Direct download only (bypass server)
    #[value(name = "direct")]
    Direct,
}
