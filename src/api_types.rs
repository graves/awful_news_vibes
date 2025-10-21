use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiEdition {
    pub local_date: String,              // "2025-10-18"
    pub time_of_day: String,             // "morning" | "afternoon" | "evening"
    pub local_time: String,              // "08:06:19.780796513" (optional precision)
    pub articles: Vec<ApiArticle>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiArticle {
    pub source: String,                  // URL string
    pub dateOfPublication: String,       // "YYYY-MM-DD"
    pub timeOfPublication: String,       // "HH:MM:SS±TZ"
    pub title: String,
    pub category: String,
    pub summaryOfNewsArticle: String,
    #[serde(default)]
    pub keyTakeAways: Vec<String>,
    #[serde(default)]
    pub namedEntities: Vec<ApiNamedEntity>,
    #[serde(default)]
    pub importantDates: Vec<ApiImportantDate>,
    #[serde(default)]
    pub importantTimeframes: Vec<ApiImportantTimeframe>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiNamedEntity {
    pub name: String,
    pub whatIsThisEntity: String,
    pub whyIsThisEntityRelevantToTheArticle: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiImportantDate {
    pub dateMentionedInArticle: String,        // "YYYY-MM-DD"
    pub descriptionOfWhyDateIsRelevant: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiImportantTimeframe {
    pub approximateTimeFrameStart: String,     // strings in API
    pub approximateTimeFrameEnd: String,
    pub descriptionOfWhyTimeFrameIsRelevant: String,
}