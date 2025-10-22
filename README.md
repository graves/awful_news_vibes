# awful_news_vibes

Daily news meta-analysis pipeline with AI-powered clustering and visualization.

## Overview

Analyzes news articles from multiple outlets, clusters related stories, extracts narrative patterns, and generates interactive D3 visualizations revealing media framing, emotional tone, and coverage dynamics.

## Features

- **Multi-source clustering**: Groups related stories across outlets using semantic similarity
- **Narrative analysis**: Detects divergence, emotional tone, and framing differences
- **Visualization suite**: 12+ interactive D3 charts (momentum, divergence, compass, word clouds, etc.)
- **Flexible LLM support**: Use different models for compression vs analysis via `awful_aj`
- **Local/remote modes**: Fetch from HTTP API or process local JSON files

## Installation

```bash
cargo build --release
```

## Configuration

Create `~/.config/awful_aj/config.yaml`:

```yaml
api_key: "YOUR_API_KEY"
api_base: "http://localhost:5001/v1"  # or OpenAI/Anthropic endpoint
model: "qwen3_30b_a3"
```

## Usage

### Basic Usage

```bash
# Fetch from HTTP API (default)
cargo run --release

# Specify output directory
cargo run --release -- --output-dir ./viz

# Use local API files (server-side)
cargo run --release -- --api-dir ./api
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output-dir <DIR>` | Output directory for generated files | `out` |
| `-c, --config <PATH>` | Path to config file (overrides `AJ_CONFIG`) | `~/.config/awful_aj/config.yaml` |
| `--cluster-config <PATH>` | Separate LLM config for cluster compression | Same as `--config` |
| `--vibe-config <PATH>` | Separate LLM config for insights/meta-analysis | Same as `--config` |
| `--api-dir <PATH>` | Local API directory (reads JSON from disk) | HTTP mode |

### Examples

```bash
# Use different LLMs for compression vs analysis
cargo run --release -- \
  --cluster-config fast-llm.yaml \
  --vibe-config slow-llm.yaml

# Process local API files
cargo run --release -- \
  --api-dir /data/news_api \
  --output-dir /var/www/viz

# Custom everything
cargo run --release -- \
  -c ~/my-llm-config.yaml \
  -o ./output \
  --api-dir ./local-cache
```

## Environment Variables

- `AJ_CONFIG`: Path to config file
- `AJ_CONFIG_DIR`: Base config directory
- `AJ_TEMPLATE_DIR`: Custom template directory
- `AJ_TEMPLATE_CLUSTER`: Cluster compression template name
- `AJ_TEMPLATE_INSIGHTS`: Insights template name
- `AJ_TEMPLATE_FINAL`: Final post template name
- `RUST_LOG`: Logging level (`info`, `debug`, `trace`)

## API Directory Structure

When using `--api-dir`, expected structure:

```
api/
├── 2025-10-21/
│   ├── morning.json
│   ├── afternoon.json
│   └── evening.json
├── 2025-10-22/
│   ├── morning.json
│   ├── afternoon.json
│   └── evening.json
```

JSON format matches `https://news.awfulsec.com/api/{YYYY-MM-DD}/{slot}.json`

## Output Structure

```
out/
├── index.json                  # Date index for visualization picker
└── 2025-10-21/
    ├── clusters.full.json      # Complete cluster data
    ├── insights.full.json      # Meta-analysis insights
    ├── meta_post.json          # Daily summary (JSON)
    ├── meta_post.md            # Daily summary (Markdown)
    ├── viz.lifecycles.json     # Story lifecycles
    ├── viz.momentum.json       # Coverage velocity
    ├── viz.divergence.json     # Narrative framing differences
    ├── viz.emotion.json        # Emotional temperature
    ├── viz.compass.json        # Topic positioning
    ├── viz.clouds.json         # Word frequency
    ├── viz.fingerprints.json   # Multi-dimensional characteristics
    ├── viz.silences.json       # Notable absences
    ├── viz.entities.json       # Entity co-mention graph
    └── viz.index.json          # Metadata
```

## Visualizations

Open `daily_analysis.html` in a browser to view:

1. **Story Momentum** - Rate of coverage change
2. **Narrative Divergence** - Outlet framing differences (blame/cause, risk/optimism)
3. **Emotional Temperature** - Anxiety/optimism/panic/ambiguity levels
4. **Story Compass** - Thematic positioning (structural/conflict/human/future)
5. **Silence Tracker** - Notable coverage gaps
6. **Story Fingerprints** - Multi-axis characteristics
7. **Word Clouds** - Outlet-specific and story-specific term frequency
8. **Story Lifecycles** - Coverage patterns (flashstorm, wildfire, resurrection, etc.)

## Architecture

### Pipeline Stages

1. **Fetch** - Retrieve or load 4 editions (yesterday morning/afternoon/evening + today morning)
2. **Deduplicate** - Remove duplicate articles across editions
3. **Cluster** - Group related stories using semantic similarity (entities, dates, tags, text)
4. **Compress** - LLM summarizes each cluster (parallel batches of 12)
5. **Insights** - Meta-analysis identifies patterns, silences, and trends
6. **Finalize** - Generate daily summary post
7. **Visualize** - Export D3-ready JSON files

### Key Components

- **fetch.rs** - HTTP API or local file loading
- **cluster.rs** - Similarity-based story grouping
- **compress.rs** - LLM compression with awful_aj
- **viz_export.rs** - D3 visualization data generation
- **orchestrator.rs** - Pipeline coordination

## Performance

- Processes 4 editions (~80-120 articles) in ~3-5 minutes
- Cluster compression: parallelized in batches of 12
- Token budget: enforced at 150k total
- ETA logging for long-running compression

## Logging

```bash
# Info level (default)
cargo run --release

# Debug level
RUST_LOG=debug cargo run --release

# Trace level (very verbose)
RUST_LOG=trace cargo run --release
```

## License

MIT

## Credits

Built with [awful_aj](https://github.com/graves/awful_aj) for LLM orchestration.
