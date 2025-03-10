import os
import streamlit as st
import pandas as pd
import json
from firecrawl import FirecrawlApp
from groq import Groq
from dotenv import load_dotenv
import time
import re

# Set page config
st.set_page_config(
    page_title="AI Freelance Job Crawler",
    page_icon="🔍",
    layout="wide"
)

# Load environment variables (for local development)
load_dotenv()

# ANSI color codes (only used for terminal output during development)
class Colors:
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

# Retrieve API keys from streamlit secrets (for deployment)
if 'FIRECRAWL_API_KEY' in st.secrets:
    firecrawl_api_key = st.secrets["FIRECRAWL_API_KEY"]
    groq_api_key = st.secrets["Groq_API_KEY"]
# Fallback to environment variables (for local development)
else:
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    groq_api_key = os.getenv("Groq_API_KEY")

# List of seed URLs for job platforms
JOB_PLATFORMS = {
    "Toptal": "https://www.toptal.com/",
    "Upwork": "https://www.upwork.com/",
    "Fiverr": "https://www.fiverr.com/",
    "Guru": "https://www.guru.com/",
    "PeoplePerHour": "https://www.peopleperhour.com/",
    "LinkedIn Jobs": "https://www.linkedin.com/jobs/",
    "Wellfound": "https://wellfound.com/",
    "Upstack": "https://upstack.co/",
    "Arc": "https://arc.dev/",
    "Turing": "https://www.turing.com/"
}

# Job search keywords and filters
JOB_KEYWORDS = [
    "data scientist freelance", 
    "data scientist contract", 
    "machine learning freelance", 
    "machine learning contract",
    "LLM freelance", 
    "LLM contract", 
    "NLP freelance", 
    "AI engineer contract",
    "remote data science contract",
    "data science consultant"
]

# URL patterns that indicate job listings
JOB_URL_PATTERNS = [
    "job", "career", "freelance", "contract", "remote", "hire", "position",
    "data-scientist", "machine-learning", "llm", "nlp", "ai-engineer"
]

def init_session_state():
    """Initialize session state variables"""
    if 'job_results' not in st.session_state:
        st.session_state.job_results = []
    if 'selected_platforms' not in st.session_state:
        st.session_state.selected_platforms = list(JOB_PLATFORMS.keys())[:3]  # Default to first 3
    if 'selected_keywords' not in st.session_state:
        st.session_state.selected_keywords = JOB_KEYWORDS.copy()
    if 'crawling_complete' not in st.session_state:
        st.session_state.crawling_complete = False
    if 'progress' not in st.session_state:
        st.session_state.progress = 0
    
    # Initialize platform checkboxes if not already set
    for platform in JOB_PLATFORMS.keys():
        checkbox_key = f"platform_{platform}"
        if checkbox_key not in st.session_state:
            # Default first 3 platforms to be selected
            st.session_state[checkbox_key] = platform in list(JOB_PLATFORMS.keys())[:3]

def initialize_clients():
    """Initialize the FirecrawlApp and Groq client if API keys are available"""
    if firecrawl_api_key and groq_api_key:
        try:
            app = FirecrawlApp(api_key=firecrawl_api_key)
            groq_client = Groq(api_key=groq_api_key)
            return app, groq_client
        except Exception as e:
            st.error(f"Error initializing clients: {str(e)}")
            return None, None
    else:
        st.error("API keys not found. Please check your Streamlit secrets or environment variables.")
        return None, None

def scrape_website(app, url, depth=2, max_urls=10):
    """
    Scrape a website using Firecrawl with job-specific configuration.
    
    Args:
        app: FirecrawlApp instance
        url (str): The base URL to scrape
        depth (int): Crawl depth
        max_urls (int): Maximum number of URLs to crawl
        
    Returns:
        dict: The scraped data
    """
    try:
        st.write(f"Scraping: {url}")
        
        # Create crawl parameters with filters for job listings
        params = {
            'formats': ['markdown', 'html'],
            'max_depth': depth,
            'max_urls': max_urls,
            'follow_links': True,
            'filters': {
                'url_contains_any': JOB_URL_PATTERNS,
                'content_contains_any': st.session_state.selected_keywords
            }
        }
        
        # Start the crawl
        scrape_result = app.scrape_url(url, params=params)
        return scrape_result
    except Exception as e:
        st.error(f"Error scraping {url}: {str(e)}")
        return None

def extract_job_listings(groq_client, content, url, platform):
    """
    Extract job listings from the scraped content using Groq's API.
    
    Args:
        groq_client: Groq client instance
        content (str): The scraped content
        url (str): The URL of the page
        platform (str): The name of the job platform
        
    Returns:
        list: Extracted job listings
    """
    try:
        st.write(f"Analyzing content from {url} for job listings...")
        
        # Create a prompt for the LLM to extract job listings
        prompt = f"""
        Analyze the following content from {platform} ({url}) and extract any freelance or contract job listings 
        related to Data Science, Machine Learning, or LLM Engineering.
        
        For each job listing found, extract the following information in JSON format:
        - job_title: The title of the job
        - job_type: The type of job (freelance, contract, remote, etc.)
        - skills_required: List of required skills
        - description: A brief description of the job
        - estimated_compensation: Any mentioned compensation (if available)
        - platform: "{platform}"
        - source_url: "{url}"
        
        Return ONLY a JSON array of job listings. If no relevant job listings are found, return an empty array.
        Don't include any introduction, explanation, or conclusion outside the JSON array.
        
        Content:
        {content[:4000]}  # Limit content to 4000 chars to avoid token limits
        """
        
        # Call Groq API to extract job listings
        completion = groq_client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",  # Use a more affordable model for this task
            messages=[
                {"role": "system", "content": "You are a specialized assistant that extracts job listings from web content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        
        # Get the response
        response_text = completion.choices[0].message.content
        
        # Extract JSON from response
        try:
            # Find JSON array in the response
            json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                job_listings = json.loads(json_str)
                return job_listings
            else:
                return []
        except Exception as json_err:
            st.warning(f"Error parsing job listings: {str(json_err)}")
            return []
    except Exception as e:
        st.error(f"Error extracting job listings: {str(e)}")
        return []

def crawl_job_platforms():
    """Crawl selected job platforms for freelance opportunities"""
    # Initialize clients
    app, groq_client = initialize_clients()
    if not app or not groq_client:
        return
    
    # Check if any platforms are selected
    if not st.session_state.selected_platforms:
        st.error("No platforms selected. Please select at least one platform to crawl.")
        return
    
    # Reset results
    st.session_state.job_results = []
    st.session_state.crawling_complete = False
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Get selected platforms
    platforms_to_crawl = {k: JOB_PLATFORMS[k] for k in st.session_state.selected_platforms}
    total_platforms = len(platforms_to_crawl)
    
    # Display which platforms are being crawled
    platform_list = ", ".join(platforms_to_crawl.keys())
    st.info(f"Crawling the following platforms: {platform_list}")
    
    try:
        # Crawl each selected platform
        for i, (platform_name, platform_url) in enumerate(platforms_to_crawl.items()):
            status_text.write(f"Crawling {platform_name}...")
            
            # For each platform, search for each selected keyword
            for keyword in st.session_state.selected_keywords:
                # Construct search URL (platform specific logic could be added here)
                search_url = platform_url
                if "upwork" in platform_url:
                    search_url = f"{platform_url}search/jobs/?q={keyword.replace(' ', '%20')}"
                elif "linkedin" in platform_url:
                    search_url = f"{platform_url}search/?keywords={keyword.replace(' ', '%20')}"
                
                # Scrape the platform with the keyword search
                scrape_result = scrape_website(app, search_url)
                
                # If successful, extract job listings
                if scrape_result and ('markdown' in scrape_result or 'html' in scrape_result):
                    content = scrape_result.get('markdown', scrape_result.get('html', ''))
                    job_listings = extract_job_listings(groq_client, content, search_url, platform_name)
                    
                    # Add to results if any listings found
                    if job_listings:
                        st.session_state.job_results.extend(job_listings)
                
                # Sleep to avoid rate limiting
                time.sleep(2)
            
            # Update progress
            progress = (i + 1) / total_platforms
            st.session_state.progress = progress
            progress_bar.progress(progress)
    except Exception as e:
        st.error(f"Error during crawling: {str(e)}")
    finally:
        # Mark crawling as complete
        st.session_state.crawling_complete = True
        status_text.write("Crawling complete!")
        progress_bar.progress(100)

def display_job_results():
    """Display the job results in a formatted way"""
    if not st.session_state.job_results:
        st.info("No job listings found. Try adjusting your search parameters.")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(st.session_state.job_results)
    
    # Add filters
    st.subheader("Filter Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'platform' in df.columns:
            platform_filter = st.multiselect(
                "Filter by Platform",
                options=sorted(df['platform'].unique()),
                default=sorted(df['platform'].unique())
            )
        else:
            platform_filter = []
    
    with col2:
        if 'job_type' in df.columns:
            job_type_filter = st.multiselect(
                "Filter by Job Type",
                options=sorted(df['job_type'].unique()),
                default=sorted(df['job_type'].unique())
            )
        else:
            job_type_filter = []
    
    # Apply filters
    filtered_df = df
    if platform_filter and 'platform' in df.columns:
        filtered_df = filtered_df[filtered_df['platform'].isin(platform_filter)]
    if job_type_filter and 'job_type' in df.columns:
        filtered_df = filtered_df[filtered_df['job_type'].isin(job_type_filter)]
    
    # Display results
    st.subheader(f"Found {len(filtered_df)} Job Listings")
    
    # Create expandable cards for each job
    for i, job in filtered_df.iterrows():
        with st.expander(f"{job.get('job_title', 'Untitled Job')} - {job.get('platform', 'Unknown Platform')}"):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"**Job Type:** {job.get('job_type', 'Not specified')}")
                st.markdown(f"**Description:** {job.get('description', 'No description available')}")
                
                # Display skills as tags
                if 'skills_required' in job and job['skills_required']:
                    skills = job['skills_required']
                    if isinstance(skills, str):
                        # Try to convert string to list if it looks like a list
                        if skills.startswith('[') and skills.endswith(']'):
                            try:
                                skills = json.loads(skills)
                            except:
                                skills = [skill.strip() for skill in skills.strip('[]').split(',')]
                        else:
                            skills = [skills]
                    
                    st.markdown("**Skills Required:**")
                    skills_html = ""
                    for skill in skills:
                        if isinstance(skill, str):
                            skills_html += f'<span style="background-color:#ddf4ff;border-radius:10px;padding:3px 8px;margin:2px;display:inline-block;font-size:0.8em">{skill}</span>'
                    st.markdown(skills_html, unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(f"**Compensation:** {job.get('estimated_compensation', 'Not specified')}")
                st.markdown(f"**Platform:** {job.get('platform', 'Unknown')}")
                if 'source_url' in job:
                    st.markdown(f"[View Job]({job['source_url']})")
    
    # Add option to download results as CSV
    if not filtered_df.empty:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="freelance_job_listings.csv",
            mime="text/csv"
        )

def build_ui():
    """Build the Streamlit UI"""
    st.title("🔍 AI Freelance Job Crawler")
    st.markdown("""
    This app crawls popular freelance platforms to find Data Science, Machine Learning, and LLM 
    contract opportunities. Select the platforms and keywords to search for.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Search Configuration")
        
        # Platform selection with more control
        st.subheader("Select Platforms to Search")
        
        # Checkboxes for each platform for more interactive selection
        platform_cols = st.columns(2)
        selected_platforms = []
        
        for i, platform in enumerate(JOB_PLATFORMS.keys()):
            col_idx = i % 2
            with platform_cols[col_idx]:
                if st.checkbox(platform, value=(i < 3), key=f"platform_{platform}"):
                    selected_platforms.append(platform)
        
        # Update session state with selected platforms
        st.session_state.selected_platforms = selected_platforms
        
        # Quick selection buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Select All"):
                for platform in JOB_PLATFORMS.keys():
                    st.session_state[f"platform_{platform}"] = True
                st.experimental_rerun()
        with col2:
            if st.button("Deselect All"):
                for platform in JOB_PLATFORMS.keys():
                    st.session_state[f"platform_{platform}"] = False
                st.experimental_rerun()
        with col3:
            if st.button("Top 3"):
                for i, platform in enumerate(JOB_PLATFORMS.keys()):
                    st.session_state[f"platform_{platform}"] = (i < 3)
                st.experimental_rerun()
        
        # Keyword selection
        st.subheader("Select Search Keywords")
        st.session_state.selected_keywords = st.multiselect(
            "Job Keywords",
            options=JOB_KEYWORDS,
            default=JOB_KEYWORDS[:3]  # Default to first 3 keywords
        )
        
        # Custom keyword input
        custom_keyword = st.text_input("Add Custom Keyword")
        if custom_keyword and st.button("Add Keyword"):
            if custom_keyword not in st.session_state.selected_keywords:
                st.session_state.selected_keywords.append(custom_keyword)
                st.experimental_rerun()
        
        # Search button
        if st.button("Search for Jobs", type="primary"):
            st.session_state.job_results = []
            st.session_state.crawling_complete = False
            st.session_state.progress = 0
    
    # Main content area
    tab1, tab2 = st.tabs(["Search", "Results"])
    
    with tab1:
        # Search tab
        st.header("Search Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Selected Platforms")
            if st.session_state.selected_platforms:
                for platform in st.session_state.selected_platforms:
                    st.write(f"- {platform}")
            else:
                st.warning("No platforms selected. Please select at least one platform from the sidebar.")
        
        with col2:
            st.subheader("Selected Keywords")
            for keyword in st.session_state.selected_keywords:
                st.write(f"- {keyword}")
        
        st.markdown("---")
        
        # Start search button (main area)
        if st.button("Start Job Search", type="primary", key="main_search"):
            if not st.session_state.selected_platforms:
                st.error("Please select at least one platform from the sidebar before starting the search.")
            else:
                with st.spinner("Crawling job platforms..."):
                    crawl_job_platforms()
        
        # Show progress if crawling
        if not st.session_state.crawling_complete and st.session_state.progress > 0:
            st.progress(st.session_state.progress)
    
    with tab2:
        # Results tab
        st.header("Job Listings Results")
        
        # If crawling is complete, display results
        if st.session_state.crawling_complete:
            display_job_results()
        elif st.session_state.job_results:
            display_job_results()
        else:
            st.info("Run a search to see results.")

def main():
    # Initialize session state
    init_session_state()
    
    # Build the UI
    build_ui()

if __name__ == "__main__":
    main()
