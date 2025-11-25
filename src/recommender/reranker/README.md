# AI Reranker

Uses Together AI to rerank GNN recommendations based on user preferences.

## Setup

source CS224W-PROJECT/bin/activate

1. Get an API key from [together.ai](https://api.together.ai/)
2. Add to `.env` in project root: `TOGETHER_API_KEY=your-key-here`
3. Run tests:
   ```
   pytest tests/reranker/ -v
   ```

