import os
import pandas as pd
import streamlit as st
from firecrawl import FirecrawlApp

# Streamlit UI
st.title("Freelance AI Job Crawler")

# Retrieve API key from Streamlit secrets
firecrawl_api_key = st.secrets["FIRECRAWL_API_KEY"]

# Initialize Firecrawl API
app = FirecrawlApp(api_key=firecrawl_api_key)

# Define job search URLs
JOB_PORTALS = [
    "https://www.upwork.com/nx/search/jobs/?q=data%20science",
    "https://www.freelancer.com/jobs/machine-learning/",
    "https://www.toptal.com/freelance-jobs",
    "https://remoteok.io/remote-ai-jobs"
]

# Define filtering keywords
JOB_KEYWORDS = ["Data Scientist", "Machine Learning", "LLM", "NLP", "AI Engineer"]
FREELANCE_KEYWORDS = ["contract", "freelance", "remote"]

def crawl_jobs():
    """Scrapes job listings using Firecrawl and filters relevant jobs."""
    job_listings = []

    for url in JOB_PORTALS:
        try:
            st.write(f"🔍 Crawling: {url}")

            # Use Firecrawl to scrape job listings
            scrape_result = app.scrape_url(url, params={'formats': ['markdown']})

            if "markdown" not in scrape_result:
                st.warning(f"⚠️ No data found on {url}")
                continue
            
            page_content = scrape_result["markdown"].lower()

            # Extract job titles from text
            for line in page_content.split("\n"):
                if any(keyword.lower() in line for keyword in JOB_KEYWORDS) and \
                   any(f_word in line for f_word in FREELANCE_KEYWORDS):
                    
                    job_listings.append({"Job Title": line.strip(), "Source": url})

        except Exception as e:
            st.error(f"Error scraping {url}: {e}")

    return job_listings

def save_to_excel(jobs):
    """Save job listings to an Excel file."""
    if not jobs:
        st.warning("⚠️ No relevant job listings found.")
        return

    df = pd.DataFrame(jobs)
    file_path = "freelance_jobs.xlsx"
    df.to_excel(file_path, index=False)

    st.success(f"✅ Job listings saved to {file_path}!")

    # Provide download link
    st.download_button("Download Excel File", data=open(file_path, "rb").read(),
                       file_name=file_path, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Streamlit UI elements
if st.button("Start Crawling"):
    with st.spinner("Crawling job listings..."):
        job_results = crawl_jobs()
        save_to_excel(job_results)
