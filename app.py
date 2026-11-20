import os
import re
import html
import requests
import gradio as gr
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from google import genai
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# ==============================================================================
# 1. Load Environment Variables & Initialize Google Gemini LLM
# ==============================================================================
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()
SCRAPEGRAPH_API_KEY = os.environ.get("SCRAPEGRAPH_API_KEY", "").strip()

if not GEMINI_API_KEY:
    raise ValueError(
        "❌ Missing GEMINI_API_KEY! Add it to your .env file or export it as an environment variable."
    )

os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
if TAVILY_API_KEY:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

gemini_llm = LLM(
    model="gemini/gemini-3.8-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.2,
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

print("✅ Gemini LLM initialized successfully!")
print(
    f"🔎 Search Engine Mode: {'Tavily API (Primary)' if TAVILY_API_KEY else 'DuckDuckGo Search (Free Auto-Fallback)'}"
)
print(
    f"🕷️ Scraper Engine Mode: {'ScrapeGraphAI (Primary)' if SCRAPEGRAPH_API_KEY else 'Smart BeautifulSoup + Gemini (Free Auto-Fallback)'}"
)

# ==============================================================================
# 2. Recursive & Semantic Chunking Module
# ==============================================================================
print("⏳ Loading local embedding model for Semantic Chunking...")
local_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def process_company_context(
    raw_context: str, strategy: str = "Semantic Chunking"
) -> str:
    """
    Splits company procurement policies and requirements into structured chunks
    using either Recursive or Semantic Chunking so agents can accurately match specs.
    """
    if not raw_context.strip():
        return "No specific company constraints provided. Prioritize best overall value and reliability."

    if strategy == "Recursive Chunking":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=350,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_text(raw_context)
    else:
        try:
            splitter = SemanticChunker(
                embeddings=local_embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=75,
            )
            chunks = splitter.split_text(raw_context)
        except Exception:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=350, chunk_overlap=50
            )
            chunks = splitter.split_text(raw_context)

    formatted_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        formatted_chunks.append(
            f"[Requirement Chunk #{idx} ({strategy})]:\n{chunk.strip()}"
        )

    return "\n\n".join(formatted_chunks)


# ==============================================================================
# 3. Custom CrewAI Tools with Automatic Free Fallbacks
# ==============================================================================
@tool("Smart_Product_Search_Tool")
def smart_product_search(query: str) -> str:
    """
    Searches the web for products, prices, and vendor URLs using Tavily Search API,
    automatically falling back to DuckDuckGo Search if Tavily is unavailable.
    """
    if TAVILY_API_KEY:
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=TAVILY_API_KEY)
            response = client.search(
                query=query, search_depth="advanced", max_results=6
            )
            results = response.get("results", [])
            if results:
                formatted = ["[Source: Tavily Search API]"]
                for r in results:
                    formatted.append(
                        f"- Title: {r.get('title')}\n  URL: {r.get('url')}\n  Snippet: {r.get('content')}"
                    )
                return "\n\n".join(formatted)
        except Exception as e:
            print(f"⚠️ Tavily failed ({e}), switching to DuckDuckGo fallback...")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=6))
            if not results:
                return "No search results found. Try broadening the search keywords."
            formatted = ["[Source: DuckDuckGo Free Search Fallback]"]
            for r in results:
                formatted.append(
                    f"- Title: {r.get('title')}\n  URL: {r.get('href')}\n  Snippet: {r.get('body')}"
                )
            return "\n\n".join(formatted)
    except Exception as e:
        return f"Search error encountered: {str(e)}"


@tool("Smart_Web_Scraper_Tool")
def smart_web_scraper(url: str) -> str:
    """
    Scrapes a product webpage and extracts structured specifications, pricing, and availability.
    Uses ScrapeGraphAI as primary engine and falls back to BeautifulSoup + Recursive Chunking + Gemini.
    """
    extraction_prompt = (
        "Extract exact product name, brand, full price with currency, key technical specifications, "
        "warranty details, customer rating, and stock availability from this webpage."
    )

    if SCRAPEGRAPH_API_KEY:
        try:
            from scrapegraph_py import Client

            sg_client = Client(api_key=SCRAPEGRAPH_API_KEY)
            response = sg_client.smartscraper(
                website_url=url, user_prompt=extraction_prompt
            )
            return f"[Source: ScrapeGraphAI] URL: {url}\nExtracted Data: {response}"
        except Exception as e:
            print(
                f"⚠️ ScrapeGraphAI failed for {url} ({e}), switching to BeautifulSoup fallback..."
            )

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        clean_text = " ".join(soup.get_text(separator=" ").split())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000, chunk_overlap=200
        )
        page_chunks = splitter.split_text(clean_text)
        relevant_content = "\n---\n".join(page_chunks[:3])

        ai_prompt = f"""
        {extraction_prompt}
        Source URL: {url}
        Webpage Content Chunks:
        {relevant_content}

        Return the extracted details clearly in bullet points. If price is not explicitly listed, state 'Price not listed on page'.
        """
        extraction = gemini_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=ai_prompt,
        )

        return f"[Source: Smart Scraper Fallback] URL: {url}\n{extraction.text}"
    except Exception as e:
        return f"[Scraping Note] Could not directly scrape {url} due to site protection ({str(e)}). Rely on search snippet data for this URL."


# ==============================================================================
# 4. CrewAI Agents, Tasks, and Sequential Execution Pipeline
# ==============================================================================
def run_vendorlens_crew(
    product_query: str, company_context: str, chunking_strategy: str
):
    """
    Executes the 4-agent sequential procurement pipeline and returns:
    1. Processed context chunks
    2. Agent execution logs / comparison summary
    3. Final clean HTML report string
    4. File path to the saved HTML report
    """
    chunked_context = process_company_context(company_context, chunking_strategy)

    search_agent = Agent(
        role="Procurement Search Specialist",
        goal="Find reputable vendors, online stores, and product listings matching the requested product and company criteria.",
        backstory=(
            "You are an expert B2B & commercial sourcing specialist. You know how to craft targeted search queries "
            "to uncover accurate product pages, pricing, and vendor options across the web."
        ),
        tools=[smart_product_search],
        llm=gemini_llm,
        verbose=True,
        allow_delegation=False,
    )

    scraper_agent = Agent(
        role="Web Data Extraction & Scraping Engineer",
        goal="Visit the discovered product URLs and extract accurate prices, technical specs, warranty terms, and availability.",
        backstory=(
            "You specialize in extracting clean, structured product specifications and pricing from e-commerce "
            "and supplier webpages while filtering out marketing noise."
        ),
        tools=[smart_web_scraper],
        llm=gemini_llm,
        verbose=True,
        allow_delegation=False,
    )

    analyst_agent = Agent(
        role="Senior Procurement & Value Analyst",
        goal="Evaluate, compare, and rank all collected products strictly against the company's chunked requirements.",
        backstory=(
            "You are a meticulous procurement analyst. You score products based on Price, Technical Specs, "
            "Compliance with Company Policy, and Overall Value-for-Money, clearly flagging any rule violations."
        ),
        llm=gemini_llm,
        verbose=True,
        allow_delegation=False,
    )

    report_agent = Agent(
        role="Executive HTML Report Designer",
        goal="Generate a complete, modern, standalone HTML procurement report with embedded CSS styling.",
        backstory=(
            "You are a corporate communications and UI specialist who transforms complex procurement comparisons "
            "into clean, executive-ready HTML reports with comparison tables, badges, and clear buying recommendations."
        ),
        llm=gemini_llm,
        verbose=True,
        allow_delegation=False,
    )

    search_task = Task(
        description=(
            f"Search for the following product: '{product_query}'.\n"
            f"Consider these company requirement chunks when searching:\n{chunked_context}\n\n"
            "Perform targeted searches to find at least 3 to 5 distinct product options or vendors with valid URLs and pricing info."
        ),
        expected_output="A structured list of 3-5 candidate products with their titles, vendor names, URLs, and initial snippet details.",
        agent=search_agent,
    )

    scraping_task = Task(
        description=(
            "Using the URLs discovered by the Search Specialist, scrape the product pages using Smart_Web_Scraper_Tool "
            "(scrape up to 4 top URLs to keep execution fast and reliable). "
            "Extract exact model names, prices, technical specifications, and stock/warranty info."
        ),
        expected_output="Detailed, structured product profiles for each scraped vendor/product including exact specs and prices.",
        agent=scraper_agent,
        context=[search_task],
    )

    analysis_task = Task(
        description=(
            f"Compare and rank all gathered products for '{product_query}' against the company's requirements:\n"
            f"{chunked_context}\n\n"
            "For each product:\n"
            "1. Check compliance with budget and mandatory specs.\n"
            "2. Assign a Value-for-Money Score (out of 10).\n"
            "3. List Pros, Cons, and Price-to-Spec trade-offs.\n"
            "4. Rank them from #1 (Best Overall Recommendation) down to the lowest."
        ),
        expected_output="A comprehensive ranked comparison analysis with scores, compliance status, and a clear #1 procurement recommendation.",
        agent=analyst_agent,
        context=[scraping_task],
    )

    report_task = Task(
        description=(
            "Create a complete, self-contained HTML5 procurement report based on the analyst's ranked comparison.\n"
            "Requirements for the HTML report:\n"
            "- Include modern internal CSS (<style>) with a clean corporate color palette (navy blue #1e3a8a, slate gray, emerald green badges).\n"
            "- Header section titled 'VendorLens AI — Executive Procurement Report' with the requested item.\n"
            "- Executive Summary & Top Recommendation highlight box.\n"
            "- Responsive Comparison Table comparing Rank, Product Name, Vendor, Price, Key Specs, Compliance Status, Score, and clickable Source Link.\n"
            "- Detailed Pros & Cons cards for each evaluated option.\n"
            "- Output ONLY raw valid HTML code starting with <!DOCTYPE html> and ending with </html>. Do not wrap in markdown backticks."
        ),
        expected_output="Valid, standalone HTML5 document string ready to be rendered and saved as .html.",
        agent=report_agent,
        context=[analysis_task],
    )

    crew = Crew(
        agents=[search_agent, scraper_agent, analyst_agent, report_agent],
        tasks=[search_task, scraping_task, analysis_task, report_task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    raw_html = str(result.raw).strip()
    clean_html = re.sub(
        r"^```html\s*|\s*```$",
        "",
        raw_html,
        flags=re.IGNORECASE | re.MULTILINE,
    ).strip()

    report_filename = "VendorLens_Procurement_Report.html"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(clean_html)

    analysis_summary = (
        str(analysis_task.output.raw)
        if analysis_task.output
        else "Analysis completed."
    )

    return chunked_context, analysis_summary, clean_html, report_filename


# ==============================================================================
# 5. Gradio Web Interface
# ==============================================================================
def execute_procurement_ui(product_query, company_context, chunking_strategy):
    """Wrapper function connecting Gradio inputs to the VendorLens AI Crew."""
    if not product_query or not product_query.strip():
        error_html = "<div style='color:#b91c1c; padding:20px; font-family:sans-serif;'>⚠️ Please enter a product name or specification to search.</div>"
        return error_html, "No input provided.", "No chunks generated.", None

    try:
        chunks_out, analysis_out, report_html, file_path = run_vendorlens_crew(
            product_query=product_query,
            company_context=company_context,
            chunking_strategy=chunking_strategy,
        )

        escaped_html = html.escape(report_html)
        iframe_preview = f"""
        <iframe srcdoc="{escaped_html}"
                style="width: 100%; height: 750px; border: 1px solid #cbd5e1; border-radius: 10px; background: white;">
        </iframe>
        """
        return iframe_preview, analysis_out, chunks_out, file_path
    except Exception as e:
        error_html = f"<div style='color:#b91c1c; padding:20px; font-family:sans-serif;'>❌ Execution Error: {str(e)}</div>"
        return error_html, str(e), "Error during chunking.", None


custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
)

with gr.Blocks(
    theme=custom_theme,
    title="VendorLens AI | Multi-Agent Procurement Assistant",
) as demo:
    gr.Markdown(
        """
        # 🔍 VendorLens AI — Multi-Agent Procurement Assistant
        **Powered by CrewAI & Google Gemini** | Automated Web Sourcing, Spec Extraction, Policy Matching & HTML Reporting
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("### 📋 1. Procurement Request & Company Policy")

            product_input = gr.Textbox(
                label="Product Name & Technical Needs",
                placeholder="e.g., Business Laptop 32GB RAM 1TB SSD Intel Core Ultra 7",
                lines=2,
            )

            company_context_input = gr.Textbox(
                label="Company Context & Procurement Rules (Budget, Warranty, Brand Preferences)",
                value=(
                    "Company: TechNova Solutions.\n"
                    "Budget Limit: Maximum $1,600 per unit.\n"
                    "Mandatory Specs: Minimum 32GB RAM, 1TB NVMe SSD, lightweight under 1.8kg.\n"
                    "Warranty & Vendor Policy: Must include at least 1-year official warranty from reputable retailers. Avoid refurbished units."
                ),
                lines=6,
            )

            chunk_strategy_input = gr.Radio(
                choices=["Semantic Chunking", "Recursive Chunking"],
                value="Semantic Chunking",
                label="Context Chunking Strategy (Requirement Processing)",
                info="Semantic splits by policy topic shifts; Recursive splits hierarchically by character length.",
            )

            run_btn = gr.Button(
                "🚀 Run Multi-Agent Procurement Crew",
                variant="primary",
                size="lg",
            )

            gr.Examples(
                examples=[
                    [
                        "Ergonomic Office Mesh Chair with Lumbar Support",
                        "Budget: Max $450 per chair. Must have adjustable armrests, breathable mesh back, and 3+ years warranty. Buying 15 units.",
                        "Semantic Chunking",
                    ],
                    [
                        "4K IPS USB-C Monitor 27 inch for Software Developers",
                        "Budget: Up to $500. Requires 65W+ USB-C power delivery, height-adjustable stand, and 99% sRGB color accuracy.",
                        "Recursive Chunking",
                    ],
                ],
                inputs=[product_input, company_context_input, chunk_strategy_input],
                label="💡 Quick Test Scenarios (Click to Load)",
            )

        with gr.Column(scale=6):
            gr.Markdown("### 📊 2. Procurement Results & Executive Report")

            download_file = gr.File(
                label="📥 Download Final HTML Procurement Report"
            )

            with gr.Tabs():
                with gr.TabItem("📄 Live HTML Report Preview"):
                    html_output = gr.HTML(
                        value="<div style='padding:40px; text-align:center; color:#64748b; border:2px dashed #cbd5e1; border-radius:10px;'>Your generated HTML procurement report will appear here after running the agents.</div>"
                    )

                with gr.TabItem("🧠 Agent Comparison & Ranking Log"):
                    analysis_output = gr.Textbox(
                        label="Senior Procurement Analyst Output",
                        lines=22,
                        interactive=False,
                    )

                with gr.TabItem("🧩 Company Context Chunks"):
                    chunks_output = gr.Textbox(
                        label="Processed Requirement Chunks Sent to Agents",
                        lines=22,
                        interactive=False,
                    )

    run_btn.click(
        fn=execute_procurement_ui,
        inputs=[product_input, company_context_input, chunk_strategy_input],
        outputs=[html_output, analysis_output, chunks_output, download_file],
    )

if __name__ == "__main__":
    demo.launch(share=False, debug=False)
