use super::args::OutputFormat;
use colored::*;
use serde_json::Value;

pub fn setup_logging(verbose: u8, quiet: bool) {
    let level = if quiet {
        // Even in quiet mode, we need tracing for the underlying libraries
        // but we'll set it to ERROR level to suppress most output
        tracing::Level::ERROR
    } else {
        match verbose {
            0 => tracing::Level::WARN,
            1 => tracing::Level::INFO,
            2 => tracing::Level::DEBUG,
            _ => tracing::Level::TRACE,
        }
    };

    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .with_writer(if quiet {
            // In quiet mode, redirect tracing to stderr to keep stdout clean
            std::io::stderr
        } else {
            std::io::stderr
        })
        .init();
}

pub fn print_output<T: serde::Serialize>(data: &T, format: &OutputFormat) {
    match format {
        OutputFormat::Human => {
            // For human format, we'll implement custom formatting per command
            // This is a fallback that shouldn't be used for complex data
            if let Ok(json_value) = serde_json::to_value(data) {
                print_human_readable(&json_value);
            }
        }
        OutputFormat::Json => {
            if let Ok(json) = serde_json::to_string(data) {
                println!("{json}");
            }
        }
        OutputFormat::JsonPretty => {
            if let Ok(json) = serde_json::to_string_pretty(data) {
                println!("{json}");
            }
        }
    }
}

pub fn print_human_readable(value: &Value) {
    match value {
        Value::Object(map) => {
            for (key, val) in map {
                match val {
                    Value::String(s) => println!("{}: {}", key.cyan().bold(), s),
                    Value::Number(n) => println!("{}: {}", key.cyan().bold(), n),
                    Value::Bool(b) => println!("{}: {}", key.cyan().bold(), b),
                    _ => println!("{}: {}", key.cyan().bold(), val),
                }
            }
        }
        _ => println!("{value}"),
    }
}
