# Job Description Summarization
## About
This project is an AI Agent that extracts Job Descriptions from the web and summarizes them. Here is the agentic flow.

![Flow](/assets/flow.png)

## Key Scripts and Usage
| Step | File Name  | Description   | Useage |
|------------|------------------------------|------------------|------------------------------------|
|1| *jd_scrape.py* | Scrapes job description from company websites and saves them in the output folder (second argument). This script requires a list of job description links to scrape. A sample input file is provided [here](/jd_scraping/jd_links_combined.csv). | ```python jd_scraping/jd_scrape.py jd_scraping/jd_links_combined.csv jd_scraping/scraped_output``` |
|2| *llm_summarization.py* | Summarizes the job descriptions into a comma separated list of skills. | ```python jd_scraping/llm_summarization.py jd_scraping/jd_links_combined.csv jd_scraping/scraped_output``` |
|3| *analysis.ipynb* | Notebook that analyzes the output of the summarization task |