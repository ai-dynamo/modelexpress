use modelexpress_client::{Client, ClientConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ClientConfig::default();
    let mut client = Client::new(config).await?;
    println!("MXClient started");

    client.send_message("Hello from MXClient!").await?;
    println!("Message sent");

    client.health_check().await?;
    println!("Health checked");

    client.initialize_nixl_agent().await?;
    println!("NIXL agent initialized");

    Ok(())
}
