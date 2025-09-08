import requests
import pandas as pd
import numpy as np
import json
from openai import OpenAI
from typing import List, Dict, Any, Optional
import time
import logging
import uuid
import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_gmap_place_id(place_name, api_key):
    """Fetch Google Maps Place ID for a given place name"""
    search_url = (
        "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
        f"?input={place_name}&inputtype=textquery&fields=place_id&key={api_key}"
    )
    search_resp = requests.get(search_url)
    data = search_resp.json()

    if not data.get("candidates"):
        return f"No place found for '{place_name}'."

    place_id = data["candidates"][0]["place_id"]
    return place_id

def fetch_google_reviews(place_id, api_key):
    """Fetch Google reviews for a given place ID"""
    details_url = (
        "https://maps.googleapis.com/maps/api/place/details/json"
        f"?place_id={place_id}&fields=name,rating,reviews&key={api_key}"
    )
    details_resp = requests.get(details_url)
    details_data = details_resp.json()

    reviews = []
    for rev in details_data.get("result", {}).get("reviews", []):
        reviews.append({
            "author": rev.get("author_name"),
            "rating": rev.get("rating"),
            "text": rev.get("text")
        })

    return reviews

class FashionFeedbackProcessor:
    """Main class for processing fashion retail feedback"""

    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model

        # Define JSON schemas for structured extraction
        self.positive_schema = {
            "sentiment": "Positive",
            "customer_name": "Extracted or 'Anonymous'",
            "store_branch": "Extracted from text or null",
            "visit_date": "Date or null",
            "brand_mentioned": "In-house brand name or null",
            "mentioned_staff": "Name of staff praised or null",
            "praised_category": "e.g., Staff Behavior, Fitting Rooms, Product Quality, Store Ambience, Pricing, Collection/Variety",
            "praised_product": "e.g., Saree, Jeans, Kurti, Lehenga, Churidar, Dupatta, Blouse, Palazzo, Anarkali, Shirt, Trousers, Skirt, Dress, Accessories, or null",
            "specific_praise": "A concise 5-7 word summary of what they liked",
            "customer_praise_quotes": [
                "exact phrase 1",
                "exact phrase 2",
                "exact phrase 3"
            ],
            "satisfaction_score": 9,
            "repeat_intent": "True/False",
            "promo_or_offer_praise": "True/False",
            "loyalty_signal": "True/False (e.g., 'I always shop here')",
            "brand_affinity_signal": "True/False (emotional attachment to brand)",
            "word_of_mouth_intent": "True/False (intends to recommend to others)",
            "purchase_expansion_intent": "True/False (mentions buying more in future)"
        }

        self.negative_schema = {
          "sentiment": "Negative",
          "customer_name": "Extracted or 'Anonymous'",
          "brand_mentioned": "brand name or null",
          "mentioned_staff": "Name of staff criticized or null",
          "complaint_category": "e.g., Staff Behavior",
          "complaint_product": "e.g., Saree",
          "specific_issue": "A concise 5-7 word summary of the problem",
          "resolution_requested": "e.g., Refund, Exchange, Replacement, Apology, or null",
          "wait_time_issue": "True/False",
          "stock_availability_issue": "True/False",
          "hygiene_or_safety_flag": "True/False",
          "lost_customer_risk": "Score 1-10 (likelihood of churn)",
          "competitor_shift_signal": "True/False (mentions switching to competitor)",
          "trust_erosion_signal": "True/False (e.g., 'Felt cheated', 'Never expected this')",
          "emotional_intensity": "Score 1-10 (calm to very angry)",
          "patience_level": "Score 1-10 (tolerant to impatient)",
          "tone_of_voice": "e.g., calm, frustrated, angry, sarcastic, disappointed, resigned",
          "perceived_value_gap": "True/False (mentions not worth price)",
          "fairness_concern": "True/False (mentions unfair treatment)",
          "expectation_gap": "True/False (expectations vs reality mismatch)",
          "forgiveness_signal": "True/False (willing to forgive if resolved)",
          "abandonment_stage": "e.g., Early frustration, Threatening exit, Already lost",
          "relationship_duration_signal": "e.g., 'First-time customer', '5-year loyalist', or null",
          "social_amplification_risk": "True/False (threatens to post publicly/social media)",
          "service_recovery_expectation": "e.g., Refund, Replacement, Apology, Staff action, or null",
          "criticality_score": 8,
          "action_to_be_performed": "e.g., Staff action"
      }

    def agent_1_sentiment_classifier(self, review_text: str) -> Dict[str, str]:
        """Classify sentiment of a review"""
        prompt = f"""Classify the sentiment of the following customer review from an Indian fashion retail store. Output ONLY a valid JSON object with the key 'sentiment' and a value of either 'Positive' or 'Negative'. Do not output any other text.
        Review: {review_text}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a sentiment analysis expert. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )

            result_text = response.choices[0].message.content.strip()
            sentiment_result = json.loads(result_text)

            # Validate the response
            if "sentiment" not in sentiment_result or sentiment_result["sentiment"] not in ["Positive", "Negative"]:
                raise ValueError("Invalid sentiment classification")

            print(f"Agent 1 classified sentiment: {sentiment_result['sentiment']}")
            return sentiment_result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Agent 1 error: {e}")
            # Fallback to neutral classification
            return {"sentiment": "Negative"}  # Default to negative for safety in business context

    def agent_2_detail_extractor(self, review_text: str, sentiment: str) -> Dict[str, Any]:
        """Extract detailed information from a review based on sentiment"""
        if sentiment == "Positive":
            schema_dict = self.positive_schema
            prompt = f"""You are an expert data analyst for a fashion retail chain. The following review is Positive. Extract information strictly into this JSON schema: {json.dumps(schema_dict, indent=2)}
            Instructions:
              - Be concise and accurate
              - Infer the satisfaction_score (1-10) from how happy the customer sounds
              - For praised_category, choose from: Staff Behavior, Fitting Rooms, Product Quality, Store Ambience, Pricing, Collection/Variety
              - For praised_product, mention specific items like: Saree, Jeans, Kurti, Lehenga, Churidar, Dupatta, Blouse, Palazzo, Anarkali, Shirt, Trousers, Skirt, Dress, Accessories
              - **customer_praise_quotes**: This is a list of direct, impactful quotes from the review. Follow these rules:
                - Extract 3-5 short, vivid phrases that describe what they loved.
                - Prioritize emotional language, adjectives, and unique turns of phrase.
                - These quotes will be used for marketing, so choose the most compelling ones.
                - **Examples of good quotes:** "softest fabric ever", "staff went above and beyond", "worth every penny", "felt like a queen", "perfect wedding outfit".
                - If the review is very generic (e.g., "Nice"), you may have fewer quotes.
              - If details are missing, use null
              - Output ONLY valid JSON, no other text

              Review: {review_text}"""

        else:  # Negative sentiment
            schema_dict = self.negative_schema
            prompt = f"""You are an expert data analyst for a fashion retail chain. The following review is Negative. Extract information strictly into this JSON schema: {json.dumps(schema_dict, indent=2)}
            Instructions:
            - Be concise and accurate
            - **complaint_category**: Choose ONLY ONE primary category from this list: [`Staff Behavior`, `Fitting Rooms`, `Product Quality - Fabric`, `Product Quality - Stitching`, `Product Quality - Size & Fit`, `Product Quality - Color`,
              `Billing & Pricing`, `Hygiene & Cleanliness`, `Exchange & Refund Policy`, `Stock Availability`, `Payment Issues`, `Delivery & Logistics`, `Car Parking Issues`, `Store Ambience & Facilities`, `Fraud & Authenticity`, 'Other']
            - **complaint_product**: Choose the MOST SPECIFIC product from this list. Use `null` if no product is mentioned, [`Saree`, `Salwar Suit`, `Kurta`, `Kurti`, `Lehenga`, `Churidar`, `Dupatta`, `Blouse`, `Palazzo`, `Anarkali`, `Sherwani`, `Dhoti`, `Jewellery`, `Bags`, `Footwear`, `Accessories`, `Dress`, `Shirt`, `Trousers`, `Jeans`, `Skirt`, `Top`, `Blazer`]
            - **criticality_score (1-10)**: Calculate based on this matrix:
              - `10`: Life/Safety risk (e.g., structural hazard, electrical fault).
              - `9`: Severe health/hygiene issue (e.g., insects in store, mold on clothes).
              - `8`: Accusation of fraud/theft; severe public shaming by staff.
              - `7`: Major financial loss (large overcharge); product ruined an event (e.g., wedding lehenga torn).
              - `6`: Strong accusation of selling counterfeits; staff extreme rudeness.
              - `5`: Standard product defect (tear, fade); billing error; denied promised exchange.
              - `4`: Minor product issue; mild staff neglect; mild inconvenience.
              - `3-1`: Subjective dislike (e.g., "didn't like the color"); very minor issues.
            - Output valid values for all the keys in the schema, DO NOT OMIT ANY KEY-VALUE PAIR.
            # FINAL INSTRUCTION: Output MUST be a perfectly parsable JSON object that validates against the provided schema. Do not add any other text, commentary, or formatting outside the JSON.

            Review: {review_text}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data extraction expert. Respond only with valid JSON that matches the exact schema provided."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )

            result_text = response.choices[0].message.content.strip()
            detail_result = json.loads(result_text)

            # Validate required fields
            if sentiment == "Positive":
                required_fields = ["sentiment", "satisfaction_score"]
            else:
                required_fields = ["sentiment", "criticality_score"]

            for field in required_fields:
                if field not in detail_result:
                    raise ValueError(f"Missing required field: {field}")

            print(f"Agent 2 extracted details for {sentiment} review")
            print('-'*60)
            return detail_result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Agent 2 error: {e}")
            # Return fallback structure
            if sentiment == "Positive":
                return {
                    "sentiment": "Positive",
                    "customer_name": "Anonymous",
                    "store_branch": None,
                    "visit_date": None,
                    "brand_mentioned": None,
                    "mentioned_staff": None,
                    "praised_category": None,
                    "praised_product": None,
                    "specific_praise": None,
                    "satisfaction_score": 5,
                    "repeat_intent": None,
                    "promo_or_offer_praise": None,
                    "loyalty_signal": None,
                    "brand_affinity_signal": None,
                    "word_of_mouth_intent": None,
                    "purchase_expansion_intent": None
                }
            else:
                return {
                    "sentiment": "Negative",
                    "customer_name": None,
                    "brand_mentioned": None,
                    "mentioned_staff": None,
                    "complaint_category": None,
                    "complaint_product": None,
                    "specific_issue": None,
                    "resolution_requested": None,
                    "wait_time_issue": None,
                    "stock_availability_issue": None,
                    "hygiene_or_safety_flag": None,
                    "lost_customer_risk": None,
                    "competitor_shift_signal": None,
                    "trust_erosion_signal": None,
                    "emotional_intensity": None,
                    "patience_level": None,
                    "tone_of_voice": None,
                    "perceived_value_gap": None,
                    "fairness_concern": None,
                    "expectation_gap": None,
                    "forgiveness_signal": None,
                    "abandonment_stage": None,
                    "relationship_duration_signal": None,
                    "social_amplification_risk": None,
                    "service_recovery_expectation": None,
                    "criticality_score": 5,
                    "action_to_be_performed": None
                }

    def process_single_review(self, review_text: str) -> Dict[str, Any]:
        """Process a single review through the pipeline"""
        # Agent 1: Sentiment Classification
        sentiment_result = self.agent_1_sentiment_classifier(review_text)
        sentiment = sentiment_result["sentiment"]

        # Small delay to respect API rate limits
        time.sleep(0.1)

        # Agent 2: Detailed Extraction
        detailed_result = self.agent_2_detail_extractor(review_text, sentiment)
        detailed_result['original_review'] = review_text

        return detailed_result

    def process_feedback_batch(self, reviews: List[str]) -> tuple[List[Dict], List[Dict], List[Dict]]:
        """Process a batch of reviews"""
        positive_reviews_dataset = []
        negative_reviews_dataset = []
        critical_reviews_dataset = []

        print(f"Processing {len(reviews)} reviews...")
        print('-'*60)

        for i, review in enumerate(reviews):
            try:
                print(f"Processing review {i+1}/{len(reviews)}")
                structured_review = self.process_single_review(review)

                # Route to appropriate dataset
                if structured_review["sentiment"] == "Positive":
                    positive_reviews_dataset.append(structured_review)
                else:
                    negative_reviews_dataset.append(structured_review)
                    if structured_review.get('criticality_score', 0) > 7:
                        critical_review = structured_review.copy()
                        critical_review['status'] = 'not processed'
                        critical_reviews_dataset.append(critical_review)

                # Rate limiting - adjust as needed based on your API limits
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error processing review {i+1}: {e}")
                continue

        print(f"Processing complete. Positive: {len(positive_reviews_dataset)}, Negative: {len(negative_reviews_dataset)}")
        print('-'*60)
        return positive_reviews_dataset, negative_reviews_dataset, critical_reviews_dataset

    def save_datasets(self, positive_reviews: List[Dict], negative_reviews: List[Dict], critical_reviews: List[Dict], output_dir: str = ".") -> None:
        """Save datasets to JSON files"""
        # Save positive reviews
        positive_file = f"{output_dir}/positive_reviews_dataset.json"
        with open(positive_file, 'w', encoding='utf-8') as f:
            json.dump(positive_reviews, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(positive_reviews)} positive reviews to {positive_file}")
        print('-'*60)

        # Save negative reviews
        negative_file = f"{output_dir}/negative_reviews_dataset.json"
        with open(negative_file, 'w', encoding='utf-8') as f:
            json.dump(negative_reviews, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(negative_reviews)} negative reviews to {negative_file}")
        print('-'*60)

        critical_file = f"{output_dir}/critical_reviews_dataset.json"
        with open(critical_file, 'w', encoding='utf-8') as f:
            json.dump(critical_reviews, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(critical_reviews)} critical reviews to {critical_file}")

# Critical Review Management Classes
class ActionType(Enum):
    """Predefined action types for critical reviews"""
    RAISE_URGENT_TICKET = "raise_urgent_ticket"
    ESCALATE_TO_MANAGER = "escalate_to_manager"
    INITIATE_REFUND = "initiate_refund"
    SCHEDULE_MAINTENANCE = "schedule_maintenance"
    SEND_APOLOGY_EMAIL = "send_apology_email"
    REQUEST_STAFF_TRAINING = "request_staff_training"
    INVENTORY_ALERT = "inventory_alert"
    QUALITY_CONTROL_ALERT = "quality_control_alert"
    CUSTOMER_CALLBACK = "customer_callback"
    STORE_VISIT_SCHEDULE = "store_visit_schedule"
    SOCIAL_MEDIA_RESPONSE = "social_media_response"
    FACILITY_REPAIR_REQUEST = "facility_repair_request"

class Priority(Enum):
    """Priority levels for actions"""
    CRITICAL = "critical"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class CriticalReview:
    """Data structure for critical review"""
    review_id: str
    store_name: str
    customer_name: str
    review_text: str
    criticality_score: float
    action_to_be_performed: str
    timestamp: datetime.datetime
    customer_contact: Optional[str] = None
    category: Optional[str] = None

@dataclass
class ActionResult:
    """Result of an automated action"""
    action_id: str
    action_type: ActionType
    status: str
    timestamp: datetime.datetime
    details: Dict[str, Any]
    follow_up_required: bool = False

# Additional classes (TicketingSystem, EmailDrafter, NotificationSystem, CriticalReviewAgent) 
# would go here - truncated for brevity but include all the classes from your original code

def process_store_list(store_list: List[str], processor: FashionFeedbackProcessor, google_maps_api_key: str):
    """Main function to process a list of stores"""
    print('Processing store list: ', store_list)
    print('-'*60)
    positive_dataset, negative_dataset, critical_dataset = [], [], []
    
    for place_name in store_list:
        print('Processing store: ', place_name)
        print('-'*60)
        
        rev_list = []
        print('-'*60)
        place_id = fetch_gmap_place_id(place_name, google_maps_api_key)
        reviews = fetch_google_reviews(place_id, google_maps_api_key)
        print('Reviews fetched for place: ', place_name)
        print('-'*60)

        for item in reviews:
            rev = item['text']
            rev_list.append(rev)

        print('Review list created.')
        print('Rows in reviews dataset: ', len(rev_list))
        print('-'*60)

        try:
            print(f"Total Reviews to be processed: {len(rev_list)}")
            positive_dataset_for_store, negative_dataset_for_store, critical_dataset_for_store = processor.process_feedback_batch(rev_list)
            
            print('[DEBUG] Adding store name to dataset.')

            for item in positive_dataset_for_store:
                item['store_name'] = place_name
            for item in negative_dataset_for_store:
                item['store_name'] = place_name
            for item in critical_dataset_for_store:
                item['store_name'] = place_name

            positive_dataset.extend(positive_dataset_for_store)
            negative_dataset.extend(negative_dataset_for_store)
            critical_dataset.extend(critical_dataset_for_store)

            print("\n" + "="*60)
            print("PROCESSING SUMMARY")
            print("="*60)
            print('-'*60)
            print(f"Positive Reviews: {len(positive_dataset)}")
            print(f"Negative Reviews: {len(negative_dataset)}")
            print(f"Critical Reviews: {len(critical_dataset)}")
            print('-'*60)

            if critical_dataset:
                print('Critical Dataset Length: ', len(critical_dataset))
                for item in critical_dataset:
                    if item.get('status') == 'not processed':
                        print('Need to process the review: ', item)
                        print('Action to be performed: ', item.get('action_to_be_performed'))

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
        
        processor.save_datasets(positive_dataset, negative_dataset, critical_dataset)

    return positive_dataset, negative_dataset, critical_dataset

# Helper functions for dashboard
def calculate_satisfaction_scores(positive_dataset):
    """Calculate average satisfaction scores by store"""
    if not positive_dataset:
        return {}
    
    store_scores = {}
    for review in positive_dataset:
        store = review.get('store_name', 'Unknown')  # Changed from store_branch
        score = review.get('satisfaction_score', 0)
        if store not in store_scores:
            store_scores[store] = []
        store_scores[store].append(score)
    
    # Calculate averages
    avg_scores = {store: sum(scores)/len(scores) for store, scores in store_scores.items()}
    return avg_scores

def calculate_dissatisfaction_scores(negative_dataset):
    """Calculate average criticality scores by store"""
    if not negative_dataset:
        return {}
    
    store_scores = {}
    for review in negative_dataset:
        store = review.get('store_name', 'Unknown')  # Changed from store_branch
        score = review.get('criticality_score', 0)
        if store not in store_scores:
            store_scores[store] = []
        store_scores[store].append(score)
    
    # Calculate averages
    avg_scores = {store: sum(scores)/len(scores) for store, scores in store_scores.items()}
    return avg_scores

def get_common_issues(negative_dataset, selected_stores):
    """Get top 5 common issues for selected stores"""
    if not negative_dataset or not selected_stores:
        return []
    
    issues = []
    for review in negative_dataset:
        if review.get('store_name') in selected_stores:  # Changed from store_branch
            issue = review.get('specific_issue')
            if issue:
                issues.append(issue)
    
    issue_counts = Counter(issues)
    return issue_counts.most_common(5)
