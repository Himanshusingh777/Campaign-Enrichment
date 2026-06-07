import json
import os
import re
import time
from urllib.parse import urlparse

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
import streamlit as st

st.set_page_config(page_title="Gemini Enrichment", layout="wide")

st.title("Gemini CSV Enrichment")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.stop()


# =====================================================
# CONFIG
# =====================================================

MODEL = "gemini-2.5-flash-lite"
DELAY_BETWEEN_CALLS = 0.5
SAVE_EVERY_ROWS = 20
CACHE_FILE = "competitor_cache.json"

OUTPUT_COMPETITOR_COL = "Suggested Competitor"
OUTPUT_COMPETITOR_CONFIDENCE_COL = "Competitor Confidence"
OUTPUT_INDUSTRY_SIGNAL_COL = "Industry Signal"
OUTPUT_INDUSTRY_CONFIDENCE_COL = "Industry Signal Confidence"
OUTPUT_STATUS_COL = "Enrichment Status"


# =====================================================
# LOAD API KEYS
# =====================================================

try:
    keys_string = st.secrets["GEMINI_API_KEYS"]
except Exception:
    load_dotenv()
    keys_string = os.getenv("GEMINI_API_KEYS", "")

API_KEYS = [key.strip() for key in keys_string.split(",") if key.strip()]

if not API_KEYS:
    st.error(
        "No Gemini API keys found.\n\n"
        "For local use: add `GEMINI_API_KEYS=key1,key2` in your `.env` file.\n\n"
        "For Streamlit Cloud: add `GEMINI_API_KEYS` in your app secrets."
    )
    st.stop()


# =====================================================
# BASIC HELPERS
# =====================================================

def clean(value, limit=500):
    if value is None:
        return ""
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    if value.lower() in ["", "nan", "none", "null", "n/a", "na", "-", "--"]:
        return ""
    return value[:limit]


def get_value(row, columns):
    for col in columns:
        if col in row.index:
            value = clean(row.get(col, ""))
            if value:
                return value
    return ""


def get_company(row):
    return get_value(row, ["Company Name for Emails", "Company Name", "Company"])


def get_website(row):
    website = get_value(row, ["Website", "Company Website"])
    if website:
        return website

    domain = get_value(row, ["DOMAIN", "Domain"])
    if domain:
        domain = domain.replace("https://", "").replace("http://", "")
        return "https://" + domain

    email = get_value(row, ["EMAIL", "Email"])
    if "@" in email:
        return "https://" + email.split("@")[-1]

    return ""


def get_domain_from_website(website):
    website = clean(website)
    if not website:
        return ""
    if not website.startswith("http://") and not website.startswith("https://"):
        website = "https://" + website
    parsed = urlparse(website)
    domain = parsed.netloc or parsed.path
    domain = domain.lower().replace("www.", "")
    domain = domain.split("/")[0]
    return domain


def normalize_company_name(name):
    name = clean(name).lower()
    name = re.sub(
        r"\b(inc|llc|ltd|limited|corp|corporation|company|co|group|holdings|solutions|technologies|technology)\b",
        "",
        name,
    )
    name = re.sub(r"[^a-z0-9]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def get_cache_key(row):
    website = get_website(row)
    domain = get_domain_from_website(website)
    if domain:
        return domain
    return normalize_company_name(get_company(row))


def load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def read_input_csv():
    return pd.read_csv(uploaded_file, dtype=str, keep_default_na=False).fillna("")


def ensure_output_columns(df):
    for col in [
        OUTPUT_COMPETITOR_COL,
        OUTPUT_COMPETITOR_CONFIDENCE_COL,
        OUTPUT_INDUSTRY_SIGNAL_COL,
        OUTPUT_INDUSTRY_CONFIDENCE_COL,
        OUTPUT_STATUS_COL,
    ]:
        if col not in df.columns:
            df[col] = ""
    return df


def save_output(df):
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# =====================================================
# API KEY ROTATION
# =====================================================

class KeyRotator:
    def __init__(self, keys):
        self.keys = keys
        self.index = 0

    def next_key(self):
        key = self.keys[self.index % len(self.keys)]
        self.index += 1
        return key


def is_quota_error(error):
    message = str(error).lower()
    return any(
        word in message
        for word in ["429", "quota", "rate", "resource_exhausted", "too many requests", "exceeded"]
    )


# =====================================================
# PROMPT
# =====================================================

def build_prompt(row):
    company = get_company(row)
    website = get_website(row)
    domain = get_domain_from_website(website)

    employees = get_value(row, ["# Employees", "Employees", "Employee Count"])
    industry = get_value(row, ["Industry"])
    keywords = get_value(row, ["Keywords"])[:900]
    technologies = get_value(row, ["Technologies"])[:500]
    revenue = get_value(row, ["Annual Revenue"])
    linkedin = get_value(row, ["Company Linkedin Url", "Company LinkedIn Url", "LinkedIn URL"])

    city = get_value(row, ["Company City", "City"])
    state = get_value(row, ["Company State", "State"])
    country = get_value(row, ["Company Country", "Country"])
    location = ", ".join([x for x in [city, state, country] if x])

    return f"""
You are enriching a cold email prospect list.

Company: {company}
Website: {website}
Domain: {domain}
Company LinkedIn: {linkedin}
Employees: {employees}
Current Broad Industry: {industry}
Keywords: {keywords}
Technologies: {technologies}
Annual Revenue: {revenue}
Location: {location}

Return JSON only.

Task 1:
Identify the company's primary specific industry signal.

Industry Signal rules:
- Be specific.
- Do not leave industry_signal blank.
- Do not use broad labels like Technology, Software, Healthcare, Finance, Business Services, IT Services, or SaaS.
- Use a specific phrase based on company, website, LinkedIn, industry, and keywords. try to keep output under 2-3 words max
- Examples:
  - Cloud Cost Optimization
  - GPS Fleet Telematics
  - Cancer Diagnostics
  - Credit Union Banking
  - Investment Consulting
  - Health Insurance Brokerage
  - Remote Patient Monitoring
  - Tax Resolution Services
  - Leave Management Software
  - Cannabis Decontamination Technology

Task 2:
Find one closest competitor.

Competitor rules:
- Competitor must solve a similar problem for a similar buyer.
- Prefer similar-size, similar-positioned, niche, regional, or mid-market competitors.
- If the company is small or mid-size, do not suggest a huge generic brand.
- Do not return the same company.
- Do not return parent company, subsidiary, partner, client, marketplace, directory, or technology vendor.
- Avoid Google, Microsoft, Amazon, Salesforce, Deloitte, Accenture, Oracle, IBM, HubSpot unless truly unavoidable.

Confidence rules:
- High = clearly same category and similar segment.
- Medium = close but not perfect.
- Low = unsure.

Return exactly this JSON:
{{
  "competitor": "",
  "competitor_confidence": "High/Medium/Low",
  "industry_signal": "",
  "industry_confidence": "High/Medium/Low"
}}

No explanation.
No markdown.
"""


# =====================================================
# GEMINI CALL
# =====================================================

def call_gemini(prompt, api_key):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=180,
            response_mime_type="application/json",
        ),
    )
    return response.text or ""


def parse_json_response(text):
    text = clean(text, 1200)
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {
                "competitor": "",
                "competitor_confidence": "Low",
                "industry_signal": "",
                "industry_confidence": "Low",
            }
        try:
            data = json.loads(match.group(0))
        except Exception:
            return {
                "competitor": "",
                "competitor_confidence": "Low",
                "industry_signal": "",
                "industry_confidence": "Low",
            }

    competitor = clean(data.get("competitor", ""), 120)
    competitor_confidence = clean(data.get("competitor_confidence", "Low"), 20).capitalize()
    industry_signal = clean(data.get("industry_signal", ""), 160)
    industry_confidence = clean(data.get("industry_confidence", "Low"), 20).capitalize()

    if competitor_confidence not in ["High", "Medium", "Low"]:
        competitor_confidence = "Low"
    if industry_confidence not in ["High", "Medium", "Low"]:
        industry_confidence = "Low"
    if not industry_signal:
        industry_signal = "Specific Industry Not Identified"
        industry_confidence = "Low"

    return {
        "competitor": competitor,
        "competitor_confidence": competitor_confidence,
        "industry_signal": industry_signal,
        "industry_confidence": industry_confidence,
    }


def enrich_row(row, key_rotator):
    prompt = build_prompt(row)
    last_error = None

    for _ in range(len(API_KEYS)):
        api_key = key_rotator.next_key()
        try:
            raw = call_gemini(prompt, api_key)
            return parse_json_response(raw)
        except Exception as error:
            last_error = error
            if is_quota_error(error):
                continue
            continue

    raise RuntimeError(f"All API keys failed. Last error: {last_error}")


# =====================================================
# MAIN
# =====================================================

def main():
    st.write(f"**Model:** {MODEL}")
    st.write(f"**API Keys Loaded:** {len(API_KEYS)}")

    df = read_input_csv()
    df = ensure_output_columns(df)

    cache = load_cache()
    key_rotator = KeyRotator(API_KEYS)

    total_rows = len(df)
    api_calls = 0
    cache_hits = 0

    progress_bar = st.progress(0)
    status_text = st.empty()
    log_area = st.empty()
    logs = []

    for index, row in df.iterrows():
        row_number = index + 1
        company = get_company(row)

        progress_bar.progress(row_number / total_rows)
        status_text.text(f"Processing row {row_number} / {total_rows}: {company or '(no company)'}")

        if not company:
            df.at[index, OUTPUT_STATUS_COL] = "Skipped - missing company"
            logs.append(f"[{row_number}] Skipped - no company name")
            log_area.text("\n".join(logs[-20:]))
            continue

        existing_competitor = clean(row.get(OUTPUT_COMPETITOR_COL, ""))
        existing_industry_signal = clean(row.get(OUTPUT_INDUSTRY_SIGNAL_COL, ""))

        if existing_competitor and existing_industry_signal:
            logs.append(f"[{row_number}] Already filled: {company}")
            log_area.text("\n".join(logs[-20:]))
            continue

        cache_key = get_cache_key(row)

        if (
            cache_key in cache
            and clean(cache[cache_key].get("industry_signal", ""))
            and clean(cache[cache_key].get("competitor", ""))
        ):
            result = cache[cache_key]
            cache_hits += 1
            status = "Done from cache"
            logs.append(f"[{row_number}] Cache hit: {company}")
        else:
            try:
                result = enrich_row(row, key_rotator)
                cache[cache_key] = result
                save_cache(cache)
                api_calls += 1
                status = "Done"
                time.sleep(DELAY_BETWEEN_CALLS)
                logs.append(
                    f"[{row_number}] Done: {company} → {result.get('competitor', '')} | {result.get('industry_signal', '')}"
                )
            except Exception as error:
                df.at[index, OUTPUT_STATUS_COL] = f"Failed - {str(error)[:150]}"
                logs.append(f"[{row_number}] FAILED: {company} | {error}")
                log_area.text("\n".join(logs[-20:]))
                continue

        df.at[index, OUTPUT_COMPETITOR_COL] = result.get("competitor", "")
        df.at[index, OUTPUT_COMPETITOR_CONFIDENCE_COL] = result.get("competitor_confidence", "Low")
        df.at[index, OUTPUT_INDUSTRY_SIGNAL_COL] = result.get("industry_signal", "")
        df.at[index, OUTPUT_INDUSTRY_CONFIDENCE_COL] = result.get("industry_confidence", "Low")
        df.at[index, OUTPUT_STATUS_COL] = status

        log_area.text("\n".join(logs[-20:]))

        if row_number % SAVE_EVERY_ROWS == 0:
            logs.append(f"Progress checkpoint at row {row_number}")

    csv_data = save_output(df)
    save_cache(cache)

    progress_bar.progress(1.0)
    status_text.text("Enrichment complete!")

    st.success("✅ Enrichment Completed!")
    st.write(f"**API Calls Used:** {api_calls}")
    st.write(f"**Cache Hits:** {cache_hits}")

    output_filename = uploaded_file.name.replace(".csv", "_Enrichment.csv")

    st.download_button(
        label="⬇️ Download Enriched CSV",
        data=csv_data,
        file_name=output_filename,
        mime="text/csv",
    )


if st.button("🚀 Start Enrichment"):
    main()