use chrono::{DateTime, Utc};
use model_express_common::models::{ModelProvider, ModelStatus};
use rusqlite::{Connection, Result as SqliteResult, params};
use std::sync::{Arc, Mutex};
use tracing::info;

/// Database model for tracking model download status
#[derive(Debug, Clone)]
pub struct ModelRecord {
    pub model_name: String,
    pub provider: ModelProvider,
    pub status: ModelStatus,
    pub created_at: DateTime<Utc>,
    pub last_used_at: DateTime<Utc>,
    pub message: Option<String>,
}

/// SQLite-based model status tracker for distributed systems
#[derive(Debug, Clone)]
pub struct ModelDatabase {
    connection: Arc<Mutex<Connection>>,
}

impl ModelDatabase {
    /// Create a new database instance and initialize the schema
    pub fn new(database_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let conn = Connection::open(database_path)?;

        // Create the models table if it doesn't exist
        conn.execute(
            r"
            CREATE TABLE IF NOT EXISTS models (
                model_name TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL,
                message TEXT
            )
            ",
            [],
        )?;

        // Create an index on last_used_at for efficient LRU queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_last_used_at ON models(last_used_at)",
            [],
        )?;

        info!("Model database initialized at: {}", database_path);

        Ok(Self {
            connection: Arc::new(Mutex::new(conn)),
        })
    }

    /// Get the status of a model
    pub fn get_status(&self, model_name: &str) -> SqliteResult<Option<ModelStatus>> {
        let conn = self.connection.lock().map_err(|_| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_LOCKED),
                Some("Rust mutex protecting database connection is poisoned".to_string()),
            )
        })?;
        let mut stmt = conn.prepare("SELECT status FROM models WHERE model_name = ?1")?;

        let mut rows = stmt.query_map([model_name], |row| {
            let status_str: String = row.get(0)?;
            Ok(status_str)
        })?;

        if let Some(row) = rows.next() {
            let status_str = row?;
            let status = match status_str.as_str() {
                "DOWNLOADING" => ModelStatus::DOWNLOADING,
                "DOWNLOADED" => ModelStatus::DOWNLOADED,
                "ERROR" => ModelStatus::ERROR,
                _ => ModelStatus::ERROR,
            };
            Ok(Some(status))
        } else {
            Ok(None)
        }
    }

    /// Get the full record for a model
    pub fn get_model_record(&self, model_name: &str) -> SqliteResult<Option<ModelRecord>> {
        let conn = self.connection.lock().map_err(|_| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_LOCKED),
                Some("Rust mutex protecting database connection is poisoned".to_string()),
            )
        })?;
        let mut stmt = conn.prepare(
            "SELECT model_name, provider, status, created_at, last_used_at, message FROM models WHERE model_name = ?1"
        )?;

        let mut rows = stmt.query_map([model_name], |row| {
            let provider_str: String = row.get(1)?;
            let status_str: String = row.get(2)?;
            let created_at_str: String = row.get(3)?;
            let last_used_at_str: String = row.get(4)?;
            let message: Option<String> = row.get(5)?;

            let provider = match provider_str.as_str() {
                "HuggingFace" => ModelProvider::HuggingFace,
                _ => ModelProvider::HuggingFace, // Default fallback
            };

            let status = match status_str.as_str() {
                "DOWNLOADING" => ModelStatus::DOWNLOADING,
                "DOWNLOADED" => ModelStatus::DOWNLOADED,
                "ERROR" => ModelStatus::ERROR,
                _ => ModelStatus::ERROR,
            };

            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .map_err(|_| {
                    rusqlite::Error::InvalidColumnType(
                        3,
                        "created_at".to_string(),
                        rusqlite::types::Type::Text,
                    )
                })?
                .with_timezone(&Utc);

            let last_used_at = DateTime::parse_from_rfc3339(&last_used_at_str)
                .map_err(|_| {
                    rusqlite::Error::InvalidColumnType(
                        4,
                        "last_used_at".to_string(),
                        rusqlite::types::Type::Text,
                    )
                })?
                .with_timezone(&Utc);

            Ok(ModelRecord {
                model_name: row.get(0)?,
                provider,
                status,
                created_at,
                last_used_at,
                message,
            })
        })?;

        if let Some(row) = rows.next() {
            Ok(Some(row?))
        } else {
            Ok(None)
        }
    }

    /// Set the status of a model, creating or updating the record
    pub fn set_status(
        &self,
        model_name: &str,
        provider: ModelProvider,
        status: ModelStatus,
        message: Option<String>,
    ) -> SqliteResult<()> {
        let conn = self.connection.lock().map_err(|_| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_LOCKED),
                Some("Rust mutex protecting database connection is poisoned".to_string()),
            )
        })?;
        let now = Utc::now();

        let provider_str = match provider {
            ModelProvider::HuggingFace => "HuggingFace",
        };

        let status_str = match status {
            ModelStatus::DOWNLOADING => "DOWNLOADING",
            ModelStatus::DOWNLOADED => "DOWNLOADED",
            ModelStatus::ERROR => "ERROR",
        };

        // Use INSERT OR REPLACE to handle both creation and updates
        conn.execute(
            r"
            INSERT OR REPLACE INTO models (model_name, provider, status, created_at, last_used_at, message)
            VALUES (?1, ?2, ?3, 
                COALESCE((SELECT created_at FROM models WHERE model_name = ?1), ?4),
                ?4, ?5)
            ",
            params![
                model_name,
                provider_str,
                status_str,
                now.to_rfc3339(),
                message
            ],
        )?;

        Ok(())
    }

    /// Update the `last_used_at` timestamp for a model
    pub fn touch_model(&self, model_name: &str) -> SqliteResult<()> {
        let conn = self.connection.lock().map_err(|_| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_LOCKED),
                Some("Rust mutex protecting database connection is poisoned".to_string()),
            )
        })?;
        let now = Utc::now();

        conn.execute(
            "UPDATE models SET last_used_at = ?1 WHERE model_name = ?2",
            params![now.to_rfc3339(), model_name],
        )?;

        Ok(())
    }

    /// Delete a model record
    pub fn delete_model(&self, model_name: &str) -> SqliteResult<()> {
        let conn = self.connection.lock().map_err(|_| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_LOCKED),
                Some("Rust mutex protecting database connection is poisoned".to_string()),
            )
        })?;
        conn.execute("DELETE FROM models WHERE model_name = ?1", [model_name])?;
        Ok(())
    }

    /// Get models ordered by last used (oldest first) - for future LRU cleanup
    pub fn get_models_by_last_used(&self, limit: Option<u32>) -> SqliteResult<Vec<ModelRecord>> {
        let conn = self.connection.lock().map_err(|_| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_LOCKED),
                Some("Rust mutex protecting database connection is poisoned".to_string()),
            )
        })?;

        let query = if let Some(limit) = limit {
            format!(
                "SELECT model_name, provider, status, created_at, last_used_at, message FROM models ORDER BY last_used_at ASC LIMIT {limit}"
            )
        } else {
            "SELECT model_name, provider, status, created_at, last_used_at, message FROM models ORDER BY last_used_at ASC".to_string()
        };

        let mut stmt = conn.prepare(&query)?;
        let rows = stmt.query_map([], |row| {
            let provider_str: String = row.get(1)?;
            let status_str: String = row.get(2)?;
            let created_at_str: String = row.get(3)?;
            let last_used_at_str: String = row.get(4)?;
            let message: Option<String> = row.get(5)?;

            let provider = match provider_str.as_str() {
                "HuggingFace" => ModelProvider::HuggingFace,
                _ => ModelProvider::HuggingFace,
            };

            let status = match status_str.as_str() {
                "DOWNLOADING" => ModelStatus::DOWNLOADING,
                "DOWNLOADED" => ModelStatus::DOWNLOADED,
                "ERROR" => ModelStatus::ERROR,
                _ => ModelStatus::ERROR,
            };

            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .map_err(|_| {
                    rusqlite::Error::InvalidColumnType(
                        3,
                        "created_at".to_string(),
                        rusqlite::types::Type::Text,
                    )
                })?
                .with_timezone(&Utc);

            let last_used_at = DateTime::parse_from_rfc3339(&last_used_at_str)
                .map_err(|_| {
                    rusqlite::Error::InvalidColumnType(
                        4,
                        "last_used_at".to_string(),
                        rusqlite::types::Type::Text,
                    )
                })?
                .with_timezone(&Utc);

            Ok(ModelRecord {
                model_name: row.get(0)?,
                provider,
                status,
                created_at,
                last_used_at,
                message,
            })
        })?;

        let mut models = Vec::new();
        for row in rows {
            models.push(row?);
        }

        Ok(models)
    }

    /// Get count of models with each status - for monitoring
    pub fn get_status_counts(&self) -> SqliteResult<(u32, u32, u32)> {
        let conn = self.connection.lock().map_err(|_| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_LOCKED),
                Some("Rust mutex protecting database connection is poisoned".to_string()),
            )
        })?;

        let mut downloading = 0u32;
        let mut downloaded = 0u32;
        let mut error = 0u32;

        let mut stmt = conn.prepare("SELECT status, COUNT(*) FROM models GROUP BY status")?;
        let rows = stmt.query_map([], |row| {
            let status: String = row.get(0)?;
            let count: u32 = row.get(1)?;
            Ok((status, count))
        })?;

        for row in rows {
            let (status, count) = row?;
            match status.as_str() {
                "DOWNLOADING" => downloading = count,
                "DOWNLOADED" => downloaded = count,
                "ERROR" => error = count,
                _ => {}
            }
        }

        Ok((downloading, downloaded, error))
    }

    /// Atomically attempt to claim a model for downloading using compare-and-swap semantics
    /// Returns the current status of the model:
    /// - If model doesn't exist, creates it with DOWNLOADING status and returns DOWNLOADING
    /// - If model exists, returns its current status without modification
    ///   This prevents race conditions in distributed environments
    pub fn try_claim_for_download(
        &self,
        model_name: &str,
        provider: ModelProvider,
    ) -> SqliteResult<ModelStatus> {
        let conn = self.connection.lock().map_err(|_| {
            rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_LOCKED),
                Some("Rust mutex protecting database connection is poisoned".to_string()),
            )
        })?;
        let now = Utc::now();

        let provider_str = match provider {
            ModelProvider::HuggingFace => "HuggingFace",
        };

        // Use INSERT OR IGNORE to atomically create the record only if it doesn't exist
        // This is our compare-and-swap operation
        let rows_affected = conn.execute(
            r#"
            INSERT OR IGNORE INTO models (model_name, provider, status, created_at, last_used_at, message)
            VALUES (?1, ?2, 'DOWNLOADING', ?3, ?3, 'Starting download...')
            "#,
            params![model_name, provider_str, now.to_rfc3339()],
        )?;

        if rows_affected > 0 {
            // We successfully inserted the record, so we claimed it for download
            Ok(ModelStatus::DOWNLOADING)
        } else {
            // Record already exists, get its current status directly
            let mut stmt = conn.prepare("SELECT status FROM models WHERE model_name = ?1")?;
            let mut rows = stmt.query_map([model_name], |row| {
                let status_str: String = row.get(0)?;
                Ok(status_str)
            })?;

            if let Some(row) = rows.next() {
                let status_str = row?;
                let status = match status_str.as_str() {
                    "DOWNLOADING" => ModelStatus::DOWNLOADING,
                    "DOWNLOADED" => ModelStatus::DOWNLOADED,
                    "ERROR" => ModelStatus::ERROR,
                    _ => ModelStatus::ERROR,
                };
                Ok(status)
            } else {
                // This should never happen, but handle it gracefully
                Err(rusqlite::Error::QueryReturnedNoRows)
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_database() -> (ModelDatabase, TempDir) {
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");
        let db_path = temp_dir.path().join("test_models.db");
        let db = ModelDatabase::new(db_path.to_str().expect("Invalid path"))
            .expect("Failed to create test database");
        (db, temp_dir)
    }

    #[test]
    fn test_database_creation() {
        let (db, _temp_dir) = create_test_database();
        // If we get here without panicking, the database was created successfully

        // Test that we can perform basic operations
        let result = db.get_status("non-existent-model");
        assert!(result.is_ok());
        assert!(result.expect("Failed to get status").is_none());
    }

    #[test]
    fn test_set_and_get_status() {
        let (db, _temp_dir) = create_test_database();
        let model_name = "test-model";
        let provider = ModelProvider::HuggingFace;
        let status = ModelStatus::DOWNLOADING;

        // Set status
        let result = db.set_status(model_name, provider, status, None);
        assert!(result.is_ok());

        // Get status
        let retrieved_status = db.get_status(model_name).expect("Failed to get status");
        assert!(retrieved_status.is_some());
        assert_eq!(retrieved_status.expect("Status should be present"), status);
    }

    #[test]
    fn test_update_status() {
        let (db, _temp_dir) = create_test_database();
        let model_name = "test-model";
        let provider = ModelProvider::HuggingFace;

        // Set initial status
        db.set_status(model_name, provider, ModelStatus::DOWNLOADING, None)
            .expect("Failed to set initial status");

        // Update status
        db.set_status(
            model_name,
            provider,
            ModelStatus::DOWNLOADED,
            Some("Success".to_string()),
        )
        .expect("Failed to update status");

        // Verify update
        let status = db
            .get_status(model_name)
            .expect("Failed to get status")
            .expect("Status should be present");
        assert_eq!(status, ModelStatus::DOWNLOADED);
    }

    #[test]
    fn test_get_full_model_record() {
        let (db, _temp_dir) = create_test_database();
        let model_name = "test-model";
        let provider = ModelProvider::HuggingFace;
        let message = "Test message";

        // Set status with message
        db.set_status(
            model_name,
            provider,
            ModelStatus::DOWNLOADED,
            Some(message.to_string()),
        )
        .expect("Failed to set status");

        // Get full record
        let record = db
            .get_model_record(model_name)
            .expect("Failed to get model record");
        assert!(record.is_some());

        let record = record.expect("Record should be present");
        assert_eq!(record.model_name, model_name);
        assert_eq!(record.provider, provider);
        assert_eq!(record.status, ModelStatus::DOWNLOADED);
        assert_eq!(
            record.message.as_ref().expect("Message should be present"),
            message
        );
    }

    #[test]
    fn test_touch_model() {
        let (db, _temp_dir) = create_test_database();
        let model_name = "test-model";
        let provider = ModelProvider::HuggingFace;

        // Create model record
        db.set_status(model_name, provider, ModelStatus::DOWNLOADED, None)
            .expect("Failed to create model record");

        // Get initial record
        let initial_record = db
            .get_model_record(model_name)
            .expect("Failed to get initial record")
            .expect("Record should be present");

        // Sleep a bit to ensure time difference
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Touch the model
        db.touch_model(model_name).expect("Failed to touch model");

        // Get updated record
        let updated_record = db
            .get_model_record(model_name)
            .expect("Failed to get updated record")
            .expect("Record should be present");

        // last_used_at should be updated
        assert!(updated_record.last_used_at > initial_record.last_used_at);
        // created_at should remain the same
        assert_eq!(updated_record.created_at, initial_record.created_at);
    }

    #[test]
    fn test_delete_model() {
        let (db, _temp_dir) = create_test_database();
        let model_name = "test-model";
        let provider = ModelProvider::HuggingFace;

        // Create model record
        db.set_status(model_name, provider, ModelStatus::DOWNLOADED, None)
            .expect("Failed to create model record");

        // Verify it exists
        assert!(
            db.get_status(model_name)
                .expect("Failed to get status")
                .is_some()
        );

        // Delete the model
        db.delete_model(model_name).expect("Failed to delete model");

        // Verify it's gone
        assert!(
            db.get_status(model_name)
                .expect("Failed to get status")
                .is_none()
        );
    }

    #[test]
    fn test_get_models_by_last_used() {
        let (db, _temp_dir) = create_test_database();
        let provider = ModelProvider::HuggingFace;

        // Create multiple models
        db.set_status("model1", provider, ModelStatus::DOWNLOADED, None)
            .expect("Failed to create model1");
        std::thread::sleep(std::time::Duration::from_millis(10));

        db.set_status("model2", provider, ModelStatus::DOWNLOADED, None)
            .expect("Failed to create model2");
        std::thread::sleep(std::time::Duration::from_millis(10));

        db.set_status("model3", provider, ModelStatus::DOWNLOADED, None)
            .expect("Failed to create model3");

        // Get all models ordered by last used
        let models = db
            .get_models_by_last_used(None)
            .expect("Failed to get models");
        assert_eq!(models.len(), 3);

        // Should be ordered by last_used_at (oldest first)
        assert_eq!(models[0].model_name, "model1");
        assert_eq!(models[1].model_name, "model2");
        assert_eq!(models[2].model_name, "model3");

        // Test with limit
        let limited_models = db
            .get_models_by_last_used(Some(2))
            .expect("Failed to get limited models");
        assert_eq!(limited_models.len(), 2);
        assert_eq!(limited_models[0].model_name, "model1");
        assert_eq!(limited_models[1].model_name, "model2");
    }

    #[test]
    fn test_get_status_counts() {
        let (db, _temp_dir) = create_test_database();
        let provider = ModelProvider::HuggingFace;

        // Initially should be all zeros
        let (downloading, downloaded, error) =
            db.get_status_counts().expect("Failed to get status counts");
        assert_eq!(downloading, 0);
        assert_eq!(downloaded, 0);
        assert_eq!(error, 0);

        // Add models with different statuses
        db.set_status("model1", provider, ModelStatus::DOWNLOADING, None)
            .expect("Failed to set model1 status");
        db.set_status("model2", provider, ModelStatus::DOWNLOADING, None)
            .expect("Failed to set model2 status");
        db.set_status("model3", provider, ModelStatus::DOWNLOADED, None)
            .expect("Failed to set model3 status");
        db.set_status("model4", provider, ModelStatus::ERROR, None)
            .expect("Failed to set model4 status");

        // Check counts
        let (downloading, downloaded, error) =
            db.get_status_counts().expect("Failed to get status counts");
        assert_eq!(downloading, 2);
        assert_eq!(downloaded, 1);
        assert_eq!(error, 1);
    }

    #[test]
    fn test_model_provider_string_conversion() {
        let (db, _temp_dir) = create_test_database();
        let model_name = "test-model";

        // Test HuggingFace provider
        db.set_status(
            model_name,
            ModelProvider::HuggingFace,
            ModelStatus::DOWNLOADED,
            None,
        )
        .expect("Failed to set status");

        let record = db
            .get_model_record(model_name)
            .expect("Failed to get record")
            .expect("Record should be present");
        assert_eq!(record.provider, ModelProvider::HuggingFace);
    }

    #[test]
    fn test_model_status_string_conversion() {
        let (db, _temp_dir) = create_test_database();
        let provider = ModelProvider::HuggingFace;

        // Test all status variants
        let statuses = [
            ModelStatus::DOWNLOADING,
            ModelStatus::DOWNLOADED,
            ModelStatus::ERROR,
        ];

        for (i, status) in statuses.iter().enumerate() {
            let model_name = format!("model{i}");
            db.set_status(&model_name, provider, *status, None)
                .expect("Failed to set status");

            let retrieved_status = db
                .get_status(&model_name)
                .expect("Failed to get status")
                .expect("Status should be present");
            assert_eq!(retrieved_status, *status);
        }
    }

    #[test]
    fn test_concurrent_access() {
        let (db, _temp_dir) = create_test_database();
        let provider = ModelProvider::HuggingFace;

        // Test that multiple operations can be performed without deadlock
        for i in 0..10 {
            let model_name = format!("model{i}");
            db.set_status(&model_name, provider, ModelStatus::DOWNLOADED, None)
                .expect("Failed to set status");
            let _status = db.get_status(&model_name).expect("Failed to get status");
            db.touch_model(&model_name).expect("Failed to touch model");
        }

        let models = db
            .get_models_by_last_used(None)
            .expect("Failed to get models");
        assert_eq!(models.len(), 10);
    }

    #[test]
    fn test_try_claim_for_download_new_model() {
        let (db, _temp_dir) = create_test_database();
        let model_name = "new-model";
        let provider = ModelProvider::HuggingFace;

        // Try to claim the model for download
        let status = db
            .try_claim_for_download(model_name, provider)
            .expect("Failed to claim for download");
        assert_eq!(status, ModelStatus::DOWNLOADING);

        // Verify the model record was created
        let record = db
            .get_model_record(model_name)
            .expect("Failed to get record")
            .expect("Record should be present");
        assert_eq!(record.model_name, model_name);
        assert_eq!(record.provider, provider);
        assert_eq!(record.status, ModelStatus::DOWNLOADING);
    }

    #[test]
    fn test_try_claim_for_download_existing_model() {
        let (db, _temp_dir) = create_test_database();
        let model_name = "existing-model";
        let provider = ModelProvider::HuggingFace;

        // Pre-create the model record as DOWNLOADED
        db.set_status(model_name, provider, ModelStatus::DOWNLOADED, None)
            .expect("Failed to set initial status");

        // Try to claim the model for download
        let status = db
            .try_claim_for_download(model_name, provider)
            .expect("Failed to claim for download");
        assert_eq!(status, ModelStatus::DOWNLOADED);

        // Verify the model record was not modified
        let record = db
            .get_model_record(model_name)
            .expect("Failed to get record")
            .expect("Record should be present");
        assert_eq!(record.model_name, model_name);
        assert_eq!(record.provider, provider);
        assert_eq!(record.status, ModelStatus::DOWNLOADED);
    }

    #[test]
    fn test_try_claim_for_download_race_condition() {
        let (db, _temp_dir) = create_test_database();
        let model_name = "race-condition-model";
        let provider = ModelProvider::HuggingFace;

        // Simulate two concurrent attempts to claim the model
        let status1 = db
            .try_claim_for_download(model_name, provider)
            .expect("Failed to claim for download 1");
        let status2 = db
            .try_claim_for_download(model_name, provider)
            .expect("Failed to claim for download 2");

        // First call should claim it (DOWNLOADING), second should see it exists (DOWNLOADING)
        assert_eq!(status1, ModelStatus::DOWNLOADING);
        assert_eq!(status2, ModelStatus::DOWNLOADING);

        // Verify the model record reflects the DOWNLOADING status
        let record = db
            .get_model_record(model_name)
            .expect("Failed to get record")
            .expect("Record should be present");
        assert_eq!(record.model_name, model_name);
        assert_eq!(record.provider, provider);
        assert_eq!(record.status, ModelStatus::DOWNLOADING);
    }

    #[test]
    fn test_try_claim_for_download_compare_and_swap() {
        let (db, _temp_dir) = create_test_database();
        let model_name = "test-cas-model";
        let provider = ModelProvider::HuggingFace;

        // First claim should succeed and return DOWNLOADING
        let status1 = db
            .try_claim_for_download(model_name, provider)
            .expect("Failed to claim for download 1");
        assert_eq!(status1, ModelStatus::DOWNLOADING);

        // Second claim should return DOWNLOADING (existing status)
        let status2 = db
            .try_claim_for_download(model_name, provider)
            .expect("Failed to claim for download 2");
        assert_eq!(status2, ModelStatus::DOWNLOADING);

        // Update to DOWNLOADED
        db.set_status(model_name, provider, ModelStatus::DOWNLOADED, None)
            .expect("Failed to update status");

        // Third claim should return DOWNLOADED (existing status)
        let status3 = db
            .try_claim_for_download(model_name, provider)
            .expect("Failed to claim for download 3");
        assert_eq!(status3, ModelStatus::DOWNLOADED);
    }
}
