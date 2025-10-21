mod models;
mod fetch;
mod similarity;
mod cluster;
mod prompts;
mod compress;
mod budget;
mod orchestrator;
mod out_models;
mod render;
mod api_types;
mod viz_export;

use anyhow::Result;
use chrono::{Datelike, Duration, Utc};
use chrono_tz::America::New_York;
use orchestrator::run_daily;
use awful_aj::{config, template};
use tracing::{debug, info, warn};
use clap::Parser;

/// Awful News Vibes - Daily news digest generator
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Output directory for generated files (default: "out")
    #[arg(short, long, default_value = "out")]
    output_dir: String,
    
    /// Path to config file (overrides AJ_CONFIG environment variable)
    #[arg(short, long)]
    config: Option<String>,
    
    /// Path to API config file for cluster compression (use different LLM)
    #[arg(long)]
    cluster_config: Option<String>,
    
    /// Path to API config file for vibe summary/insights (use different LLM)
    #[arg(long)]
    vibe_config: Option<String>,
}


fn resolve_paths() -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf)> {
    // 1) Base config dir — prefer env override, else awful_aj::config_dir()
    let base_dir = if let Ok(dir) = std::env::var("AJ_CONFIG_DIR") {
        std::path::PathBuf::from(dir)
    } else {
        awful_aj::config_dir()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
    };

    // 2) Config file — prefer AJ_CONFIG, else <base>/config.yaml
    let cfg_path = if let Ok(p) = std::env::var("AJ_CONFIG") {
        std::path::PathBuf::from(p)
    } else {
        base_dir.join("config.yaml")
    };

    // 3) Template dir — prefer AJ_TEMPLATE_DIR, else <base>/templates
    let tpl_dir = if let Ok(p) = std::env::var("AJ_TEMPLATE_DIR") {
        std::path::PathBuf::from(p)
    } else {
        let d = base_dir.join("templates");
        // make it visible to awful_aj::template loader
        std::env::set_var("AJ_TEMPLATE_DIR", &d);
        d
    };

    Ok((base_dir, cfg_path, tpl_dir))
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
        )
        .with_target(false)
        .with_thread_ids(false)
        .with_line_number(true)
        .init();

    info!("Starting awful_news_vibes");

    // Parse command line arguments first
    let args = Args::parse();

    // Determine config path: CLI arg > resolve_paths logic
    let cfg_path = if let Some(ref config_path) = args.config {
        debug!("Using config file from --config argument: {}", config_path);
        std::path::PathBuf::from(config_path)
    } else {
        let (_base_dir, cfg_path, _tpl_dir) = resolve_paths()?;
        debug!("Using config file from environment/default: {}", cfg_path.display());
        cfg_path
    };

    // Friendlier error if missing
    if !cfg_path.exists() {
        return Err(anyhow::anyhow!(
            "awful_aj config not found at {}\n\
             Use --config to specify a config file, or set AJ_CONFIG environment variable.\n\
             Example config.yaml:\n\
             api_key: \"YOUR_KEY\"\napi_base: \"http://localhost:5001/v1\"\nmodel: \"qwen3_30b_a3\"\n",
            cfg_path.display()
        ));
    }

    // Load config via awful_aj
    let cfg = config::load_config(
        cfg_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("invalid config path"))?,
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    // Load templates by name; loader uses AJ_TEMPLATE_DIR (set above) or defaults
    let tpl_cluster_name  = std::env::var("AJ_TEMPLATE_CLUSTER").unwrap_or_else(|_| "meta_vibe_cluster_compressor".to_string());
    let tpl_insights_name = std::env::var("AJ_TEMPLATE_INSIGHTS").unwrap_or_else(|_| "meta_vibe_insights".to_string());
    let tpl_final_name    = std::env::var("AJ_TEMPLATE_FINAL").unwrap_or_else(|_| "meta_vibe_daily_post".to_string());

    let tpl_cluster  = template::load_template(&tpl_cluster_name).await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let tpl_insights = template::load_template(&tpl_insights_name).await.map_err(|e| anyhow::anyhow!(e.to_string()))?;
    let tpl_final    = template::load_template(&tpl_final_name).await.map_err(|e| anyhow::anyhow!(e.to_string()))?;

    // Date calculations - using Eastern time to match news.awfulsec.com timezone
    let utc_now = Utc::now();
    let eastern_now = utc_now.with_timezone(&New_York);
    let today = eastern_now.date_naive();
    let yesterday = today - Duration::days(1);

    let ymd_today = format!("{:04}-{:02}-{:02}", today.year(), today.month(), today.day());
    let ymd_yesterday = format!("{:04}-{:02}-{:02}", yesterday.year(), yesterday.month(), yesterday.day());
    
    info!(
        "Date range - yesterday={}, today={}, output_dir={}",
        ymd_yesterday, ymd_today, args.output_dir
    );
    debug!("Using Eastern timezone - current_time={}", eastern_now.format("%Y-%m-%d %H:%M:%S %Z"));

    // Load optional API configs for different LLM endpoints
    let cluster_cfg = if let Some(ref cluster_cfg_path) = args.cluster_config {
        debug!("Loading cluster API config from: {}", cluster_cfg_path);
        config::load_config(cluster_cfg_path).map_err(|e| anyhow::anyhow!(e.to_string()))?
    } else {
        debug!("Using default API config for cluster compression");
        cfg.clone()
    };

    let vibe_cfg = if let Some(ref vibe_cfg_path) = args.vibe_config {
        debug!("Loading vibe API config from: {}", vibe_cfg_path);
        config::load_config(vibe_cfg_path).map_err(|e| anyhow::anyhow!(e.to_string()))?
    } else {
        debug!("Using default API config for vibe summary/insights");
        cfg.clone()
    };

    run_daily(&cfg, &cluster_cfg, &vibe_cfg, &tpl_cluster, &tpl_insights, &tpl_final, &ymd_yesterday, &ymd_today, &args.output_dir).await
}
