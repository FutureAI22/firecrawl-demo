# AI Freelance Job Crawler

An app that crawls popular job platforms to find Data Science, Machine Learning, and LLM contract opportunities.

## Features

- Search across multiple job platforms including LinkedIn, Indeed, Glassdoor, and more
- Add custom platforms to search
- Search for specific keywords related to data science and AI roles
- Extract and display relevant job listings
- Export search results as JSON

## Setup

### Local Development

1. Clone this repository
2. Create a `.env` file with your API keys:
```
FIRECRAWL_API_KEY=your_firecrawl_api_key
Groq_API_KEY=your_groq_api_key
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Run the app:
```
streamlit run app.py
```

### Deployment on Streamlit Cloud

1. Push this repository to GitHub
2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create a new app pointing to your GitHub repository
4. Add your API keys in the Streamlit Cloud secrets management:
   - Go to your app settings
   - Find the "Secrets" section
   - Add the following:
   ```
   FIRECRAWL_API_KEY=your_firecrawl_api_key
   Groq_API_KEY=your_groq_api_key
   ```
5. Deploy your app

## API Requirements

This app requires:
- A Firecrawl API key for web scraping
- A Groq API key for content analysis

## Usage

1. Select the job platforms you want to search from the sidebar
2. Add any custom platforms if needed
3. Select or add search keywords
4. Click "Search for Jobs" to start the search
5. View the results organized by platform and keyword
6. Export the results as JSON if desired
