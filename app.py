import os
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

# Streamlit UI
st.title("Freelance Job Listings Crawler")

# Define target job roles and filtering keywords
JOB_KEYWORDS = [
    "Data Scientist", "Machine Learning", "LLM", "NLP",
    "Machine Learning Contract", "LLM Engineer"
]
FREELANCE_KEYWORDS = ["contract", "freelance", "remote"]

# Define job search URLs (example sources)
JOB_PORTALS = [
    "https://www.upwork.com/nx/search/jobs/?q=data%20science",
    "https://www.freelancer.com/jobs/machine-learning/",
    "https://www.toptal.com/freelance-jobs",
    "https://remoteok.io/remote-ai-jobs"
]

def crawl_jobs():
    """Scrape job listings from multiple sources and filter relevant jobs."""
    job_listings = []

    for url in JOB_PORTALS:
        try:
            st.write(f"🔍 Crawling: {url}")

            # Send HTTP request
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            response.raise_for_status()

            # Parse HTML content
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract job postings (Modify selectors as per website structure)
            job_elements = soup.find_all(["h2", "h3", "a", "div"])  # Common job title tags

            for job in job_elements:
                job_text = job.get_text(strip=True).lower()

                # Check if job matches keywords and freelance filters
                if any(keyword.lower() in job_text for keyword in JOB_KEYWORDS) and \
                   any(filter_word in job_text for filter_word in FREELANCE_KEYWORDS):

                    job_title = job.get_text(strip=True)
                    job_link = job.find("a", href=True)
                    job_url = job_link["href"] if job_link else url

                    job_listings.append({"Title": job_title, "URL": job_url})

        except Exception as e:
            st.error(f"Error crawling {url}: {e}")

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
