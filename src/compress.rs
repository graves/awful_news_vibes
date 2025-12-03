use anyhow::{anyhow, Result};
use awful_aj::{api::ask, config::AwfulJadeConfig, template::ChatTemplate};
use tracing::{debug, info};

pub async fn llm_call(cfg: &AwfulJadeConfig, tpl: &ChatTemplate, user: &str) -> Result<String> {
    let start = std::time::Instant::now();
    
    debug!("LLM call starting - prompt_length={} chars", user.len());
    
    // Map Box<dyn StdError> -> anyhow::Error *before* `?`
    let answer = ask(cfg, user.to_string(), tpl, None, None, false)
        .await
        .map_err(|e| anyhow!(e.to_string()))?;
    
    let elapsed = start.elapsed();
    info!(
        "LLM API call completed - duration={:.2}s, response_length={} chars",
        elapsed.as_secs_f32(),
        answer.len()
    );
    
    Ok(answer)
}
