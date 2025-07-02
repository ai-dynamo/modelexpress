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
        let conn = self.connection.lock().unwrap();
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
        let conn = self.connection.lock().unwrap();
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
        let conn = self.connection.lock().unwrap();
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
        let conn = self.connection.lock().unwrap();
        let now = Utc::now();

        conn.execute(
            "UPDATE models SET last_used_at = ?1 WHERE model_name = ?2",
            params![now.to_rfc3339(), model_name],
        )?;

        Ok(())
    }

    /// Delete a model record
    pub fn delete_model(&self, model_name: &str) -> SqliteResult<()> {
        let conn = self.connection.lock().unwrap();
        conn.execute("DELETE FROM models WHERE model_name = ?1", [model_name])?;
        Ok(())
    }

    /// Get models ordered by last used (oldest first) - for future LRU cleanup
    pub fn get_models_by_last_used(&self, limit: Option<u32>) -> SqliteResult<Vec<ModelRecord>> {
        let conn = self.connection.lock().unwrap();

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
        let conn = self.connection.lock().unwrap();

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_database() -> (ModelDatabase, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_models.db");
        let db = ModelDatabase::new(db_path.to_str().unwrap()).unwrap();
        (db, temp_dir)
    }

    #[test]
    fn test_database_creation() {
        let (db, _temp_dir) = create_test_database();
        // If we get here without panicking, the database was created successfully

        // Test that we can perform basic operations
        let result = db.get_status("non-existent-model");
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
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
        let retrieved_status = db.get_status(model_name).unwrap();
        assert!(retrieved_status.is_some());
        assert_eq!(retrieved_status.unwrap(), status);
    }

    #[test]
    fn test_update_status() {
        let (db, _temp_dir) = create_test_database();
        let model_name = "test-model";
        let provider = ModelProvider::HuggingFace;

        // Set initial status
        db.set_status(model_name, provider, ModelStatus::DOWNLOADING, None)
            .unwrap();

        // Update status
        db.set_status(
            model_name,
            provider,
            ModelStatus::DOWNLOADED,
            Some("Success".to_string()),
        )
        .unwrap();

        // Verify update
        let status = db.get_status(model_name).unwrap().unwrap();
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
        .unwrap();

        // Get full record
        let record = db.get_model_record(model_name).unwrap();
        assert!(record.is_some());

        let record = record.unwrap();
        assert_eq!(record.model_name, model_name);
        assert_eq!(record.provider, provider);
        assert_eq!(record.status, ModelStatus::DOWNLOADED);
        assert_eq!(record.message.as_ref().unwrap(), message);
    }

    #[test]
    fn test_touch_model() {
        let (db, _temp_dir) = create_test_database();
        let model_name = "test-model";
        let provider = ModelProvider::HuggingFace;

        // Create model record
        db.set_status(model_name, provider, ModelStatus::DOWNLOADED, None)
            .unwrap();

        // Get initial record
        let initial_record = db.get_model_record(model_name).unwrap().unwrap();

        // Sleep a bit to ensure time difference
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Touch the model
        db.touch_model(model_name).unwrap();

        // Get updated record
        let updated_record = db.get_model_record(model_name).unwrap().unwrap();

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
            .unwrap();

        // Verify it exists
        assert!(db.get_status(model_name).unwrap().is_some());

        // Delete the model
        db.delete_model(model_name).unwrap();

        // Verify it's gone
        assert!(db.get_status(model_name).unwrap().is_none());
    }

    #[test]
    fn test_get_models_by_last_used() {
        let (db, _temp_dir) = create_test_database();
        let provider = ModelProvider::HuggingFace;

        // Create multiple models
        db.set_status("model1", provider, ModelStatus::DOWNLOADED, None)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));

        db.set_status("model2", provider, ModelStatus::DOWNLOADED, None)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));

        db.set_status("model3", provider, ModelStatus::DOWNLOADED, None)
            .unwrap();

        // Get all models ordered by last used
        let models = db.get_models_by_last_used(None).unwrap();
        assert_eq!(models.len(), 3);

        // Should be ordered by last_used_at (oldest first)
        assert_eq!(models[0].model_name, "model1");
        assert_eq!(models[1].model_name, "model2");
        assert_eq!(models[2].model_name, "model3");

        // Test with limit
        let limited_models = db.get_models_by_last_used(Some(2)).unwrap();
        assert_eq!(limited_models.len(), 2);
        assert_eq!(limited_models[0].model_name, "model1");
        assert_eq!(limited_models[1].model_name, "model2");
    }

    #[test]
    fn test_get_status_counts() {
        let (db, _temp_dir) = create_test_database();
        let provider = ModelProvider::HuggingFace;

        // Initially should be all zeros
        let (downloading, downloaded, error) = db.get_status_counts().unwrap();
        assert_eq!(downloading, 0);
        assert_eq!(downloaded, 0);
        assert_eq!(error, 0);

        // Add models with different statuses
        db.set_status("model1", provider, ModelStatus::DOWNLOADING, None)
            .unwrap();
        db.set_status("model2", provider, ModelStatus::DOWNLOADING, None)
            .unwrap();
        db.set_status("model3", provider, ModelStatus::DOWNLOADED, None)
            .unwrap();
        db.set_status("model4", provider, ModelStatus::ERROR, None)
            .unwrap();

        // Check counts
        let (downloading, downloaded, error) = db.get_status_counts().unwrap();
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
        .unwrap();

        let record = db.get_model_record(model_name).unwrap().unwrap();
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
            db.set_status(&model_name, provider, *status, None).unwrap();

            let retrieved_status = db.get_status(&model_name).unwrap().unwrap();
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
                .unwrap();
            let _status = db.get_status(&model_name).unwrap();
            db.touch_model(&model_name).unwrap();
        }

        let models = db.get_models_by_last_used(None).unwrap();
        assert_eq!(models.len(), 10);
    }
}
