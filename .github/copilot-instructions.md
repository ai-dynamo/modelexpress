# Repository Coding Standards

- In this repository, the use of `unwrap` in Rust code is **strictly forbidden** except in benchmarks. Using expect is allowed in tests. Always handle errors using proper error handling patterns such as `match`, `?`, or custom error types.
- All pull requests will be reviewed for accidental use of `unwrap`, and shouldn't pass clippy checks. Trying to circumvent these rules except in benchmark code will result in immediate rejection of the pull request.
- All cargo dependencies should be in the root `Cargo.toml` file, and any dependency for the sub crates should be workspace exclusively.

# General Guidelines

- Follow idiomatic Rust practices as outlined in the Rust Book.
- Prefer descriptive error messages when returning errors.

# Testing

- Write tests for all new features and bug fixes.
- Use the built-in testing framework and follow the conventions outlined in the Rust Book.
- Aim for high test coverage, but prioritize testing critical functionality.
- Always try running the ./run_integration_tests.sh script at the root of the repository before claiming work is done.
- Always run `cargo test` to ensure all tests pass before claiming work is done.
- Always run `cargo clippy` to catch common mistakes and improve the code quality, and fix any warnings or errors it reports.

# Architecture

The project is split in 3 separate crates:
1. `client`: Contains a test client, a CLI tool, and a client library.
2. `common`: Provides shared utilities and data structures for the model. Any constant definitions should be placed here. As much as possible, any shared logic should also be placed here.
3. `server`: Implements the server-side logic and API endpoints for ModelExpress in a stand alone server.

# Code quality

- Do **NOT** use emojis. These are unprofessional.
- Do not create markdown files to document code changes or decisions.
- Do not over-comment code. Removing code is fine without adding new comments to explain why.
