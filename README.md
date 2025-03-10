# AI Freelance Job Crawler

This Streamlit app crawls popular freelance platforms to find Data Science, Machine Learning, and LLM contract opportunities.

## Features

- **Platform-Specific Crawling**: Crawl 10 popular job platforms like Toptal, Upwork, LinkedIn and more
- **Target Job Categories**: Focus on data science, machine learning, and LLM freelance contracts
- **Advanced Filtering**: Filter job listings by platform, job type, and other criteria
- **Error Handling**: Robust error handling for API limitations and rate limiting
- **Sample Results**: Generates sample results when sites block web crawlers

## Setup

### Prerequisites

- Python 3.7+
- Streamlit
- Firecrawl API Key
- Groq API Key

### Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API keys:
   - For local development: Create a `.env` file with:
     ```
     FIRECRAWL_API_KEY=your_key_here
     Groq_API_KEY=your_key_here
     ```
   - For Streamlit Cloud: Add these as secrets in the Streamlit Cloud dashboard

4. Run the app locally:
   ```
   streamlit run app.py
   ```

## Usage

1. Select which job platforms to crawl from the sidebar
2. Choose or add custom keywords for your job search
3. Click "Start Job Search" to begin the crawling process
4. View results in the Results tab
5. Filter results by platform, job type, or exclude sample results
6. Download results as CSV for further analysis

## Known Limitations

- Some platforms like LinkedIn and PeoplePerHour have stronger anti-crawling measures
- The app generates sample results when websites block web crawlers
- API rate limits may restrict the number of searches you can perform

## Troubleshooting

- If you encounter API errors, try reducing the number of platforms and keywords
- Use the "View Error Details" expander to see specific error messages
- For persistent API issues, try increasing the delay between requests in the code

## License

This project is licensed under the MIT License - see the LICENSE file for details.
