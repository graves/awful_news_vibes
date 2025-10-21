use anyhow::{Context, Result};
use reqwest::Client;
use xxhash_rust::xxh3::xxh3_64;
use tracing::{debug, info, warn};

use crate::api_types::*;
use crate::models::*;

fn make_cluster_article_id(url: &str, title: &str) -> String {
    format!("{:016x}", xxh3_64(format!("{}|{}", url, title).as_bytes()))
}

/// Try to fetch one edition; return Ok(None) on 404 (missing edition)
pub async fn fetch_edition_opt(client: &Client, ymd: &str, slot: &str) -> Result<Option<Edition>> {
    let url = format!("https://news.awfulsec.com/api/{}/{}.json", ymd, slot);
    let start = std::time::Instant::now();
    
    debug!("Fetching edition - date={}, slot={}", ymd, slot);
    
    let resp = client.get(&url).send().await
        .with_context(|| format!("Request failed for {}", url))?;

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        warn!("Edition not found (404) - {}/{}", ymd, slot);
        return Ok(None);
    }

    let resp = resp.error_for_status()
        .with_context(|| format!("HTTP error for {}", url))?;

    let api_ed: ApiEdition = resp.json().await
        .with_context(|| format!("Decoding JSON for {}", url))?;

    // Use the parameters we requested, not the API's metadata (which may be incorrect)
    let edition_id = format!("{}-{}", ymd, slot);
    let published_at = format!("{}T{}", api_ed.local_date, api_ed.local_time);

    debug!(
        "Edition metadata - id={}, api_date={}, api_time={}", 
        edition_id, api_ed.local_date, api_ed.time_of_day
    );
    
    let article_count = api_ed.articles.len();
    let articles: Vec<Article> = api_ed.articles.into_iter().map(|a| {
        Article {
            id: make_cluster_article_id(&a.source, &a.title),
            title: a.title,
            source: a.source,
            category: a.category,
            tags: a.tags,
            key_takeaways: a.keyTakeAways,
            named_entities: a.namedEntities.into_iter().map(|e| NamedEntity {
                name: e.name,
                kind: "UNKNOWN".to_string(),
                context: format!("{} | {}", e.whatIsThisEntity, e.whyIsThisEntityRelevantToTheArticle),
            }).collect(),
            important_dates: a.importantDates.into_iter().map(|d| DatedCtx {
                date: d.dateMentionedInArticle,
                context: d.descriptionOfWhyDateIsRelevant,
            }).collect(),
            important_timeframes: a.importantTimeframes.into_iter().map(|t| SpanCtx {
                span: format!("{}..{}", t.approximateTimeFrameStart, t.approximateTimeFrameEnd),
                context: t.descriptionOfWhyTimeFrameIsRelevant,
            }).collect(),
        }
    }).collect();

    let elapsed = start.elapsed();
    info!(
        "Edition API fetch completed - edition={}/{}, duration={:.2}s, articles={}",
        ymd, slot, elapsed.as_secs_f32(), article_count
    );

    Ok(Some(Edition {
        edition_id,
        published_at,
        articles,
    }))
}

pub fn normalize_editions(mut eds: Vec<Edition>) -> Vec<Edition> {
    for ed in eds.iter_mut() {
        for a in ed.articles.iter_mut() {
            a.title = a.title.trim().to_string();
        }
    }
    eds
}
