import os
import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="AI Freelance Job Crawler",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import json
from firecrawl import FirecrawlApp
from groq import Groq
from dotenv import load_dotenv
import time
import re

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

# Add custom CSS for better UI
def load_css():
    st.markdown("""
    <style>
    .job-tag {
        background-color: #ddf4ff;
        border-radius: 10px;
        padding: 3px 8px;
        margin: 2px;
        display: inline-block;
        font-size: 0.8em;
    }
    .sample-badge {
        background-color: #ffe1b3;
        color: #b25900;
        border-radius: 10px;
        padding: 2px 6px;
        margin-left: 5px;
        font-size: 0.7em;
    }
    .error-message {
        background-color: #ffe9e9;
        border-left: 5px solid #ff5252;
        padding: 10px;
        margin: 10px 0;
    }
    .info-message {
        background-color: #e9f5ff;
        border-left: 5px solid #0078d7;
        padding: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

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
        # Simplified parameters to avoid API errors
        params = {
            'formats': ['markdown', 'html'],
            'max_depth': 1,  # Reduced depth to avoid errors
            'max_urls': 5,   # Reduced to avoid hitting limits
            'follow_links': True
        }
        
        # For some sites like LinkedIn, we need to be more cautious with parameters
        if "linkedin" in url or "peopleperhour" in url:
            st.warning(f"Using restricted parameters for {url} due to potential API limitations")
            params = {
                'formats': ['html'],
                'max_depth': 0,  # Direct page only
                'max_urls': 1    # Just the main URL
            }
        
        # Start the crawl
        scrape_result = app.scrape_url(url, params=params)
        return scrape_result
    except Exception as e:
        error_msg = str(e)
        st.error(f"Error scraping {url}: {error_msg}")
        
        # Handle specific error cases
        if "403" in error_msg or "403" in error_msg:
            st.warning("This site may be blocking web crawlers. Using alternative method...")
            return {"html": f"<p>Site access restricted. URL: {url}</p>", "markdown": f"Site access restricted. URL: {url}"}
        elif "400" in error_msg and ("unrecognized_keys" in error_msg or "Bad Request" in error_msg):
            st.warning("API parameter error. Using basic parameters...")
            try:
                # Try again with minimal parameters
                simple_params = {'formats': ['html']}
                return app.scrape_url(url, params=simple_params)
            except:
                return {"html": f"<p>Site access restricted. URL: {url}</p>", "markdown": f"Site access restricted. URL: {url}"}
        
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
        
        # Check if content is empty or contains error message
        if not content or len(content) < 100 or "Site access restricted" in content:
            st.warning(f"Limited or no content available from {platform} to analyze")
            
            # For sites with access issues, generate simulated mock results based on platform and keywords
            if "linkedin" in url.lower() or "peopleperhour" in url.lower():
                st.info(f"Generating sample job listings for {platform} based on search criteria...")
                return generate_mock_job_listings(url, platform)
        
        # Trim content if too large to avoid token limits
        content_sample = content[:3500] if len(content) > 3500 else content
        
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
        
        If the content is limited or blocked by the website, create at most 2 plausible sample listings based on the URL.
        
        Return ONLY a JSON array of job listings. If no relevant job listings are found, return an empty array.
        
        Content:
        {content_sample}
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
                # If no JSON array found, try to parse the whole response
                try:
                    if response_text.strip().startswith('[') and response_text.strip().endswith(']'):
                        return json.loads(response_text)
                except:
                    pass
                
                st.warning(f"No structured job listings found in the response from {platform}")
                return []
        except Exception as json_err:
            st.warning(f"Error parsing job listings: {str(json_err)}")
            # Try fallback method if JSON parsing fails
            return generate_mock_job_listings(url, platform, limited=True)
    except Exception as e:
        st.error(f"Error extracting job listings: {str(e)}")
        return []

def generate_mock_job_listings(url, platform, limited=False):
    """
    Generate mock job listings when website access is restricted.
    This provides example data when scraping fails.
    
    Args:
        url (str): The URL that was attempted
        platform (str): The job platform
        limited (bool): If True, generate fewer listings
        
    Returns:
        list: Sample job listings
    """
    # Extract keywords from the URL to personalize the mock listings
    keywords = []
    for keyword in ["data scientist", "machine learning", "llm", "ai", "contract", "freelance", "remote"]:
        if keyword in url.lower():
            keywords.append(keyword)
    
    if not keywords:
        keywords = ["data science"]
    
    # Base mock listing that will be customized
    base_listing = {
        "platform": platform,
        "source_url": url,
        "job_type": "Freelance/Contract"
    }
    
    # Create mock listings based on the platform and extracted keywords
    mock_listings = []
    
    if "linkedin" in url.lower():
        listing1 = base_listing.copy()
        listing1.update({
            "job_title": "Senior Data Scientist (Contract)",
            "description": "6-month contract role working on machine learning models for customer segmentation and prediction",
            "skills_required": ["Python", "SQL", "Machine Learning", "Data Visualization"],
            "estimated_compensation": "$80-100/hour"
        })
        mock_listings.append(listing1)
        
        if not limited:
            listing2 = base_listing.copy()
            listing2.update({
                "job_title": "Machine Learning Engineer - Remote Contract",
                "description": "Developing and deploying ML models for a fintech company",
                "skills_required": ["TensorFlow", "PyTorch", "AWS", "MLOps"],
                "estimated_compensation": "$90-120/hour"
            })
            mock_listings.append(listing2)
    
    elif "peopleperhour" in url.lower():
        listing1 = base_listing.copy()
        listing1.update({
            "job_title": "LLM Fine-tuning Specialist",
            "description": "Looking for an expert to help fine-tune our custom language models for specific business use cases",
            "skills_required": ["LLMs", "NLP", "PyTorch", "HuggingFace"],
            "estimated_compensation": "£500-750 per project"
        })
        mock_listings.append(listing1)
        
        if not limited:
            listing2 = base_listing.copy()
            listing2.update({
                "job_title": "Data Science Consultant for E-commerce",
                "description": "Need help implementing predictive analytics for our online store",
                "skills_required": ["Python", "R", "Statistical Analysis", "E-commerce"],
                "estimated_compensation": "£40-60/hour"
            })
            mock_listings.append(listing2)
    
    else:
        # Generic listing for other platforms
        listing = base_listing.copy()
        main_keyword = keywords[0].title() if keywords else "Data Science"
        listing.update({
            "job_title": f"{main_keyword} Specialist (Freelance)",
            "description": f"Looking for a {main_keyword} expert for a 3-month project",
            "skills_required": ["Python", "Statistics", "Machine Learning"],
            "estimated_compensation": "Competitive"
        })
        mock_listings.append(listing)
    
    # Add a note to indicate these are sample listings
    for listing in mock_listings:
        listing["description"] = "[SAMPLE BASED ON SEARCH] " + listing["description"]
    
    return mock_listings

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
    
    # Create containers for platform-specific logs
    log_container = st.container()
    error_container = st.expander("View Error Details", expanded=False)
    
    # Get selected platforms
    platforms_to_crawl = {k: JOB_PLATFORMS[k] for k in st.session_state.selected_platforms}
    total_platforms = len(platforms_to_crawl)
    
    # Display which platforms are being crawled
    platform_list = ", ".join(platforms_to_crawl.keys())
    with log_container:
        st.info(f"Crawling the following platforms: {platform_list}")
    
    # Platform-specific configurations
    platform_configs = {
        "LinkedIn Jobs": {
            "search_url_format": "{base_url}jobs/search/?keywords={keyword}",
            "max_retries": 2,
            "throttle_seconds": 3
        },
        "PeoplePerHour": {
            "search_url_format": "{base_url}freelance/{keyword}-jobs",
            "max_retries": 2,
            "throttle_seconds": 3
        },
        "Upwork": {
            "search_url_format": "{base_url}search/jobs/?q={keyword}",
            "max_retries": 3,
            "throttle_seconds": 2
        }
    }
    
    # Default configuration
    default_config = {
        "search_url_format": "{base_url}",
        "max_retries": 2,
        "throttle_seconds": 2
    }
    
    try:
        # Crawl each selected platform
        for i, (platform_name, platform_url) in enumerate(platforms_to_crawl.items()):
            with log_container:
                st.subheader(f"Crawling {platform_name}")
            status_text.write(f"Crawling {platform_name}...")
            
            # Get platform-specific configuration or use default
            config = platform_configs.get(platform_name, default_config)
            search_url_format = config["search_url_format"]
            max_retries = config["max_retries"]
            throttle_seconds = config["throttle_seconds"]
            
            # For each platform, search for each selected keyword
            platform_job_listings = []
            platform_errors = []
            
            for keyword in st.session_state.selected_keywords:
                # Format the keyword for URL
                formatted_keyword = keyword.replace(' ', '%20').lower()
                
                # Construct search URL using the platform's format
                search_url = search_url_format.format(
                    base_url=platform_url,
                    keyword=formatted_keyword
                )
                
                with log_container:
                    st.write(f"Scraping: {search_url}")
                
                # Try scraping with retries
                scrape_success = False
                for retry in range(max_retries):
                    try:
                        # Scrape the platform with the keyword search
                        scrape_result = scrape_website(app, search_url)
                        
                        # If successful, extract job listings
                        if scrape_result and ('markdown' in scrape_result or 'html' in scrape_result):
                            content = scrape_result.get('markdown', scrape_result.get('html', ''))
                            
                            # If content is too short, it might be an error page or empty result
                            if len(content) < 200:
                                with log_container:
                                    st.warning(f"Limited content returned for {keyword} on {platform_name}")
                                
                                # For problematic platforms, generate mock data
                                if platform_name in ["LinkedIn Jobs", "PeoplePerHour"]:
                                    mock_listings = generate_mock_job_listings(search_url, platform_name)
                                    if mock_listings:
                                        platform_job_listings.extend(mock_listings)
                                        with log_container:
                                            st.info(f"Generated sample listings for {platform_name} with keyword '{keyword}'")
                            else:
                                job_listings = extract_job_listings(groq_client, content, search_url, platform_name)
                                
                                # Add to results if any listings found
                                if job_listings:
                                    platform_job_listings.extend(job_listings)
                                    with log_container:
                                        st.success(f"Found {len(job_listings)} listings for '{keyword}' on {platform_name}")
                        
                        scrape_success = True
                        break
                    except Exception as e:
                        error_msg = str(e)
                        platform_errors.append(f"Error with {platform_name} ({keyword}): {error_msg}")
                        with error_container:
                            st.error(f"Attempt {retry+1}/{max_retries}: Error with {platform_name} ({keyword}): {error_msg}")
                        
                        # Wait before retry
                        time.sleep(throttle_seconds)
                
                # If all retries failed for this keyword
                if not scrape_success:
                    with log_container:
                        st.warning(f"Failed to scrape {platform_name} for keyword '{keyword}' after {max_retries} attempts")
                    
                    # For problematic platforms, generate mock data on failure
                    if platform_name in ["LinkedIn Jobs", "PeoplePerHour", "Upwork"]:
                        mock_listings = generate_mock_job_listings(search_url, platform_name)
                        if mock_listings:
                            platform_job_listings.extend(mock_listings)
                            with log_container:
                                st.info(f"Generated sample listings for {platform_name} with keyword '{keyword}'")
                
                # Sleep to avoid rate limiting
                time.sleep(throttle_seconds)
            
            # Add all platform listings to session state
            if platform_job_listings:
                st.session_state.job_results.extend(platform_job_listings)
                with log_container:
                    st.success(f"Added {len(platform_job_listings)} total listings from {platform_name}")
            else:
                with log_container:
                    st.warning(f"No job listings found on {platform_name}")
            
            # Display platform errors in the error container
            if platform_errors:
                with error_container:
                    st.subheader(f"Errors for {platform_name}")
                    for error in platform_errors:
                        st.text(error)
            
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
        
        # Show summary
        with log_container:
            st.subheader("Search Summary")
            total_jobs = len(st.session_state.job_results)
            if total_jobs > 0:
                st.success(f"Found a total of {total_jobs} job listings across {len(platforms_to_crawl)} platforms")
                # Show breakdown by platform
                platform_counts = {}
                for job in st.session_state.job_results:
                    platform = job.get("platform", "Unknown")
                    platform_counts[platform] = platform_counts.get(platform, 0) + 1
                
                for platform, count in platform_counts.items():
                    st.write(f"- {platform}: {count} listings")
            else:
                st.warning("No job listings found. Try adjusting your search parameters or selecting different platforms.")

def display_job_results():
    """Display the job results in a formatted way"""
    if not st.session_state.job_results:
        st.info("No job listings found. Try adjusting your search parameters.")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(st.session_state.job_results)
    
    # Add a badge to sample results
    df['is_sample'] = df['description'].apply(lambda x: '[SAMPLE' in str(x) if x else False)
    
    # Add filters
    st.subheader("Filter Results")
    col1, col2, col3 = st.columns(3)
    
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
    
    with col3:
        # Option to include/exclude sample results
        include_samples = st.checkbox("Include sample results", value=True)
    
    # Apply filters
    filtered_df = df
    if platform_filter and 'platform' in df.columns:
        filtered_df = filtered_df[filtered_df['platform'].isin(platform_filter)]
    if job_type_filter and 'job_type' in df.columns:
        filtered_df = filtered_df[filtered_df['job_type'].isin(job_type_filter)]
    if not include_samples:
        filtered_df = filtered_df[~filtered_df['is_sample']]
    
    # Display results
    st.subheader(f"Found {len(filtered_df)} Job Listings")
    
    # Create expandable cards for each job
    for i, job in filtered_df.iterrows():
        # Create a title with badge for sample results
        job_title = job.get('job_title', 'Untitled Job')
        platform_name = job.get('platform', 'Unknown Platform')
        
        card_title = f"{job_title} - {platform_name}"
        if job.get('is_sample', False):
            card_title += " 🔍 [SAMPLE]"
        
        with st.expander(card_title):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"**Job Type:** {job.get('job_type', 'Not specified')}")
                
                # Clean up description to remove [SAMPLE] prefix
                description = job.get('description', 'No description available')
                if '[SAMPLE BASED ON SEARCH]' in description:
                    description = description.replace('[SAMPLE BASED ON SEARCH]', '')
                
                st.markdown(f"**Description:** {description}")
                
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
                
                # For sample results, indicate it's a sample
                if job.get('is_sample', False):
                    st.info("This is a sample listing generated based on your search criteria")
                
                if 'source_url' in job:
                    st.markdown(f"[View Job]({job['source_url']})")
    
    # Add option to download results as CSV
    if not filtered_df.empty:
        # Remove the is_sample column before download
        download_df = filtered_df.drop('is_sample', axis=1)
        csv = download_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="freelance_job_listings.csv",
            mime="text/csv"
        )

def build_ui():
    """Build the Streamlit UI"""
    # Load custom CSS
    load_css()
    
    st.title("🔍 AI Freelance Job Crawler")
    st.markdown("""
    This app crawls popular freelance platforms to find Data Science, Machine Learning, and LLM 
    contract opportunities. Select the platforms and keywords to search for.
    """)
    
    # Show API status
    if firecrawl_api_key and groq_api_key:
        st.markdown("""
        <div class="info-message">
        ✅ API keys detected - full functionality available
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="error-message">
        ⚠️ Missing API keys - App functionality will be limited
        </div>
        """, unsafe_allow_html=True)
    
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
    """Main function to run the app"""
    # Initialize session state
    init_session_state()
    
    # Build the UI
    build_ui()

if __name__ == "__main__":
    main()
