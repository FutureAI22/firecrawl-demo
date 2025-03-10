import os
import re
import json
import streamlit as st
from firecrawl import FirecrawlApp
from groq import Groq
from dotenv import load_dotenv

# Load environment variables (for local development)
load_dotenv()

# App title and description
st.set_page_config(page_title="AI Freelance Job Crawler", page_icon="🔍", layout="wide")
st.title("🔍 AI Freelance Job Crawler")
st.markdown("This app crawls popular job platforms to find Data Science, Machine Learning, and LLM contract opportunities. Select the platforms and keywords to search for.")

# Initialize API clients
# First try to get API keys from Streamlit secrets (for deployed app)
try:
    firecrawl_api_key = st.secrets["FIRECRAWL_API_KEY"]
    groq_api_key = st.secrets["Groq_API_KEY"]
except Exception:
    # Fall back to environment variables (for local development)
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    groq_api_key = os.getenv("Groq_API_KEY")

# Initialize clients if keys are available
if firecrawl_api_key and groq_api_key:
    app = FirecrawlApp(api_key=firecrawl_api_key)
    groq_client = Groq(api_key=groq_api_key)
else:
    st.error("API keys not found. Please set up your API keys in Streamlit secrets or environment variables.")
    st.stop()

# Define the job platforms
job_platforms = {
    "LinkedIn": "linkedin.com",
    "Indeed": "indeed.com",
    "Glassdoor": "glassdoor.com",
    "DataScienceJobs": "datasciencejobs.com",
    "Wellfound": "wellfound.com",
    "Dice": "dice.com",
    "FlexJobs": "flexjobs.com",
    "Interview Query": "interviewquery.com"
}

# Sidebar for search configuration
st.sidebar.header("Search Configuration")

# Platform selection
st.sidebar.subheader("Select Platforms to Search")
selected_platforms = {}
for platform_name, platform_url in job_platforms.items():
    selected_platforms[platform_name] = st.sidebar.checkbox(f"{platform_name} ({platform_url})", value=False)

# Custom platform input
st.sidebar.subheader("Add Custom Platform")
custom_platform_name = st.sidebar.text_input("Platform Name")
custom_platform_url = st.sidebar.text_input("Platform URL")
if st.sidebar.button("Add Custom Platform"):
    if custom_platform_name and custom_platform_url:
        # Ensure URL has protocol
        if not custom_platform_url.startswith('http'):
            custom_platform_url = 'https://' + custom_platform_url
        job_platforms[custom_platform_name] = custom_platform_url
        selected_platforms[custom_platform_name] = True
        st.sidebar.success(f"Added {custom_platform_name}")
    else:
        st.sidebar.error("Please enter both platform name and URL")

# Keyword selection
st.sidebar.subheader("Search Keywords")
default_keywords = ["data scientist", "machine learning engineer", "data engineer", "AI engineer", "LLM developer", 
                    "data analytics", "business BI", "data visualisation", "Power BI"]
selected_keywords = []

for keyword in default_keywords:
    if st.sidebar.checkbox(keyword, value=False):
        selected_keywords.append(keyword)

# Custom keyword input
custom_keyword = st.sidebar.text_input("Add Custom Keyword")
if st.sidebar.button("Add Keyword"):
    if custom_keyword and custom_keyword not in selected_keywords:
        selected_keywords.append(custom_keyword)
        st.sidebar.success(f"Added '{custom_keyword}' to search keywords")

# Platform selection buttons
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("Select All", key="select_all_platforms"):
        for platform in job_platforms:
            selected_platforms[platform] = True
with col2:
    if st.button("Deselect All", key="deselect_all_platforms"):
        for platform in job_platforms:
            selected_platforms[platform] = False
with col3:
    if st.button("Top 3", key="top_3_platforms"):
        top_platforms = ["LinkedIn", "Indeed", "Wellfound"]
        for platform in job_platforms:
            selected_platforms[platform] = platform in top_platforms

# Search button
search_clicked = st.sidebar.button("Search for Jobs", type="primary")

# Main content area - Search Configuration Display
st.header("Search Configuration")

# Display selected platforms and keywords
col1, col2 = st.columns(2)

with col1:
    st.subheader("Selected Platforms")
    selected_platform_list = [platform for platform, selected in selected_platforms.items() if selected]
    
    if selected_platform_list:
        for platform in selected_platform_list:
            st.markdown(f"• {platform}")
    else:
        st.info("No platforms selected. Please select at least one platform from the sidebar.")

with col2:
    st.subheader("Selected Keywords")
    if selected_keywords:
        for keyword in selected_keywords:
            st.markdown(f"• {keyword}")
    else:
        st.info("No keywords selected. Please select at least one keyword from the sidebar.")

# Functions for job searching and processing
def scrape_website(url, keyword):
    """
    Scrape a website using Firecrawl, focused on the given keyword.
    
    Args:
        url (str): The URL to scrape
        keyword (str): The keyword to focus on
        
    Returns:
        dict: The scraped data
    """
    try:
        # Add search parameter to the URL if possible
        search_url = f"{url}/jobs?q={keyword.replace(' ', '+')}"
        st.info(f"Scraping: {search_url}")
        
        scrape_result = app.scrape_url(search_url, params={'formats': ['markdown']})
        return scrape_result
    except Exception as e:
        st.error(f"Error scraping {url}: {str(e)}")
        return None

def extract_job_listings(content, keyword, model="deepseek-r1-distill-llama-70b"):
    """
    Extract job listings from the scraped content using Groq's API.
    
    Args:
        content (str): The scraped content
        keyword (str): The keyword used for searching
        model (str): The model to use for extraction
        
    Returns:
        list: The extracted job listings
    """
    try:
        st.info(f"Extracting job listings for '{keyword}'...")
        
        prompt = f"""
        Extract job listings related to '{keyword}' from the following website content.
        Only extract jobs that are specifically related to data science, machine learning, AI, or closely related fields.
        
        For each job listing, extract:
        1. Job title
        2. Company name
        3. Location (if available)
        4. Job type (full-time, contract, etc. if available)
        5. A short description or requirements summary (if available)
        6. URL or link to the job posting (if available)
        
        Return the listings as a JSON array of objects with these fields. Only include jobs that are clearly related to '{keyword}'.
        
        Content:
        {content[:10000]}  # Limiting content length for API constraints
        """
        
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that specializes in extracting job listings from web content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        
        response_text = completion.choices[0].message.content
        
        # Extract JSON from response
        try:
            # Look for JSON array in the response
            json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                job_listings = json.loads(json_str)
                return job_listings
            else:
                st.warning("Could not parse JSON response. Raw output displayed instead.")
                return [{"error": "Parsing error", "raw_content": response_text}]
        except Exception as json_err:
            st.warning(f"Error parsing JSON response: {str(json_err)}")
            return [{"error": "JSON parsing error", "raw_content": response_text}]
            
    except Exception as e:
        st.error(f"Error extracting job listings: {str(e)}")
        return None

# Search execution and results display
if search_clicked:
    # Validate search inputs
    active_platforms = [platform for platform, selected in selected_platforms.items() if selected]
    
    if not active_platforms:
        st.error("Please select at least one platform to search.")
    elif not selected_keywords:
        st.error("Please select at least one keyword to search for.")
    else:
        # Create a results section
        st.header("Search Results")
        
        # Progress bar
        progress_bar = st.progress(0)
        total_searches = len(active_platforms) * len(selected_keywords)
        search_count = 0
        
        # Store all results
        all_results = {}
        
        # Search each platform for each keyword
        for platform in active_platforms:
            platform_url = job_platforms[platform]
            if not platform_url.startswith('http'):
                platform_url = 'https://' + platform_url
                
            platform_results = {}
            
            for keyword in selected_keywords:
                search_count += 1
                progress_bar.progress(search_count / total_searches)
                
                st.subheader(f"Searching {platform} for '{keyword}'")
                
                # Scrape the website
                scrape_result = scrape_website(platform_url, keyword)
                
                if scrape_result and 'markdown' in scrape_result:
                    content = scrape_result['markdown']
                    
                    # Extract job listings
                    job_listings = extract_job_listings(content, keyword)
                    
                    if job_listings and len(job_listings) > 0:
                        platform_results[keyword] = job_listings
                        
                        # Display results for this keyword
                        st.write(f"Found {len(job_listings)} potential matches for '{keyword}' on {platform}:")
                        
                        for i, job in enumerate(job_listings):
                            with st.expander(f"{i+1}. {job.get('job_title', 'Unnamed Position')} - {job.get('company_name', 'Unknown Company')}"):
                                st.write(f"**Job Title:** {job.get('job_title', 'N/A')}")
                                st.write(f"**Company:** {job.get('company_name', 'N/A')}")
                                st.write(f"**Location:** {job.get('location', 'N/A')}")
                                st.write(f"**Job Type:** {job.get('job_type', 'N/A')}")
                                
                                if 'description' in job and job['description']:
                                    st.write("**Description:**")
                                    st.write(job['description'])
                                
                                if 'url' in job and job['url']:
                                    st.write(f"**Link:** [{job['url']}]({job['url']})")
                    else:
                        st.info(f"No job listings found for '{keyword}' on {platform}.")
                else:
                    st.warning(f"Failed to scrape {platform} for '{keyword}'.")
            
            # Store results for this platform
            if platform_results:
                all_results[platform] = platform_results
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        
        # Summary of results
        st.header("Search Summary")
        total_jobs = sum(len(results) for platform_data in all_results.values() 
                         for results in platform_data.values())
        
        st.write(f"Found a total of {total_jobs} potential job matches across {len(active_platforms)} platforms.")
        
        # Save results button
        if total_jobs > 0:
            if st.button("Export Results (JSON)"):
                # Convert results to JSON
                results_json = json.dumps(all_results, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=results_json,
                    file_name="job_search_results.json",
                    mime="application/json"
                )
else:
    st.info("Configure your search parameters in the sidebar and click 'Search for Jobs' to start.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses Firecrawl for web scraping and Groq for content analysis. "
    "Results are meant to be a starting point for your job search and may not be comprehensive."
)
