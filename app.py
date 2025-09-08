import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from collections import Counter
import json
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

import requests
import numpy as np
import json
from openai import OpenAI
from typing import List, Dict, Any, Optional
import time
import logging
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client and other variables
client = None
api_key = None
processor = None

def process_store_list(store_list):
  print('Processing store list: ', store_list)
  print('-'*60)
  positive_dataset, negative_dataset, critical_dataset = [], [], []
  for place_name in store_list:
    print('Processing store: ', place_name)
    print('-'*60)
    
    rev_list = []
    print('-'*60)
    reviews = fetch_google_reviews(fetch_gmap_place_id(place_name), api_key)
    print('Reviews fetched for place: ', place_name)
    print('-'*60)
    # print(reviews)

    for item in reviews:
      rev = item['text']
      rev_list.append(rev)

    print('Review list created.')
    print('Rows in reviews dataset: ', len(rev_list))
    print('-'*60)


    try:
        print(f"Total Reviews to be processed: {len(rev_list)}")
        positive_dataset_for_store, negative_dataset_for_store, critical_dataset_for_store = processor.process_feedback_batch(rev_list)
        
        # positive_dataset_for_store['store_name'] = place_name
        # negative_dataset_for_store['store_name'] = place_name
        # critical_dataset_for_store['store_name'] = place_name
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

        # processor.save_datasets(positive_dataset, negative_dataset, critical_dataset)

        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        # print(f"Total Reviews Processed: {len(rev_list)}")
        print('-'*60)
        print(f"Positive Reviews: {len(positive_dataset)}")
        print(f"Negative Reviews: {len(negative_dataset)}")
        print(f"Critical Reviews: {len(critical_dataset)}")
        print('-'*60)

        if positive_dataset:
            print('Positive Dataset Length: ', len(positive_dataset))
            # print(f"\nSample Positive Review:")
            # print(json.dumps(positive_dataset[0], indent=2, ensure_ascii=False))

        if negative_dataset:
            print('Negative Dataset Length: ', len(negative_dataset))
            # print(f"\nSample Negative Review:")
            # print(json.dumps(negative_dataset[0], indent=2, ensure_ascii=False))
        if critical_dataset:
            print('Critical Dataset Length: ', len(critical_dataset))
            for item in critical_dataset:
              if item['status'] == 'not processed':
                print('Need to process the review: ', item)
                print('Action to be performed: ', item['action_to_be_performed'])
            # print(f"\nSample Critical Review:")
            # print(json.dumps(critical_dataset[0], indent=2, ensure_ascii=False))

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
    processor.save_datasets(positive_dataset, negative_dataset, critical_dataset)

  return positive_dataset, negative_dataset, critical_dataset

class FashionFeedbackProcessor:

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):

        # openai.api_key = openai_api_key
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
        prompt = f"""Classify the sentiment of the following customer review from an Indian fashion retail store. Output ONLY a valid JSON object with the key 'sentiment' and a value of either 'Positive' or 'Negative'. Do not output any other text.
        Review: {review_text}"""

        try:
            response = client.chat.completions.create(
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
            # FINAL INSTRUCTION: Output MUST be a perfec tly parsable JSON object that validates against the provided schema. Do not add any other text, commentary, or formatting outside the JSON.

            Review: {review_text}"""

        try:
            response = client.chat.completions.create(
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

        # Agent 1: Sentiment Classification
        sentiment_result = self.agent_1_sentiment_classifier(review_text)
        sentiment = sentiment_result["sentiment"]

        # Small delay to respect API rate limits
        time.sleep(0.1)

        # Agent 2: Detailed Extraction
        detailed_result = self.agent_2_detail_extractor(review_text, sentiment)
        detailed_result['original_review'] = review_text
        # detailed_result['original_review'] = review_text

        return detailed_result
    

    def process_feedback_batch(self, reviews: List[str]) -> tuple[List[Dict], List[Dict]]:

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
                    if structured_review['criticality_score'] > 7:
                      critical_review = structured_review
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

import json
import uuid
import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class TicketingSystem:
    """Mock ticketing system integration"""
    
    def __init__(self):
        self.tickets = {}
        self.ticket_counter = 1000
    
    def create_ticket(self, title: str, description: str, priority: Priority, 
                     store_name: str, category: str) -> str:
        """Create a new support ticket"""
        ticket_id = f"TICKET-{self.ticket_counter}"
        self.ticket_counter += 1
        
        ticket = {
            "id": ticket_id,
            "title": title,
            "description": description,
            "priority": priority.value,
            "store_name": store_name,
            "category": category,
            "status": "open",
            "created_at": datetime.datetime.now(),
            "assigned_to": self._assign_ticket(priority, category)
        }
        
        self.tickets[ticket_id] = ticket
        logger.info(f"Created ticket {ticket_id} with priority {priority.value}")
        return ticket_id

    def _assign_ticket(self, priority: Priority, category: str) -> str:
        """Auto-assign tickets based on category and priority"""
        if priority == Priority.CRITICAL:
            return "senior_manager@zudio.com"
        elif category in ["product_quality", "inventory"]:
            return "operations_manager@zudio.com"
        elif category in ["staff_behavior", "service"]:
            return "hr_manager@zudio.com"
        else:
            return "store_manager@zudio.com"

class EmailDrafter:
    """Email drafting and sending system"""
    
    def __init__(self):
        self.email_templates = {
            "apology": self._get_apology_template(),
            "escalation": self._get_escalation_template(),
            "training_request": self._get_training_template(),
            "maintenance": self._get_maintenance_template()
        }
    
    def draft_email(self, template_type: str, context: Dict[str, Any]) -> Dict[str, str]:
        """Draft an email based on template and context"""
        template = self.email_templates.get(template_type, "")
        
        email_draft = {
            "to": context.get("recipient", ""),
            "subject": template["subject"].format(**context),
            "body": template["body"].format(**context),
            "priority": context.get("priority", "normal"),
            "draft_id": f"DRAFT-{uuid.uuid4().hex[:8]}"
        }
        
        logger.info(f"Drafted {template_type} email: {email_draft['draft_id']}")
        return email_draft
    
    def _get_apology_template(self):
        return {
            "subject": "We're Sorry - Immediate Action on Your Feedback at {store_name}",
            "body": """Dear {customer_name},

We sincerely apologize for the poor experience you had at our {store_name} location. 
Your feedback is extremely valuable to us, and we take all concerns seriously.

Issue Identified: {issue_summary}
Immediate Actions Taken:
- Ticket #{ticket_id} has been created and assigned to our senior management
- Store management has been notified for immediate attention
- We are implementing corrective measures to prevent similar issues

We would like to make this right. Our customer service team will contact you within 24 hours 
to discuss how we can resolve this matter to your satisfaction.

Thank you for bringing this to our attention.

Best regards,
Customer Experience Team
Zudio Fashion
"""
        }
    
    def _get_escalation_template(self):
        return {
            "subject": "URGENT: Critical Customer Issue - {store_name} - Action Required",
            "body": """Dear {recipient_name},

A critical customer issue (Severity Score: {criticality_score}) has been identified at {store_name} 
requiring immediate management attention.

Customer: {customer_name}
Issue Category: {category}
Review Summary: {review_summary}
Recommended Action: {action_to_be_performed}

This issue has been automatically flagged due to its high severity. Please review and take 
immediate action within 4 hours.

Ticket Reference: {ticket_id}
Automated Actions Taken: {automated_actions}

Best regards,
Automated Review Management System
"""
        }
    
    def _get_training_template(self):
        return {
            "subject": "Staff Training Request - {store_name} - {category}",
            "body": """Dear HR Team,

Based on recent customer feedback analysis, we've identified a need for additional staff training 
at {store_name} in the following area: {category}

Issue Details: {issue_summary}
Recommended Training Focus: {training_focus}
Staff Members to Include: {staff_list}

Please schedule appropriate training sessions within the next 2 weeks.

Training Request ID: {request_id}
Priority: {priority}

Best regards,
Quality Assurance Team
"""
        }
    
    def _get_maintenance_template(self):
        return {
            "subject": "Facility Maintenance Request - {store_name} - {issue_type}",
            "body": """Dear Maintenance Team,

A facility maintenance issue has been identified at {store_name} based on customer feedback.

Issue: {issue_description}
Location: {store_name}
Priority: {priority}
Customer Impact: High

Please schedule maintenance within 48 hours to resolve this issue.

Maintenance Request ID: {request_id}
Contact: {store_manager_contact}

Best regards,
Facility Management System
"""
        }

class NotificationSystem:
    """Handle various notification channels"""
    
    def send_sms_alert(self, phone_number: str, message: str) -> bool:
        """Send SMS alert (mock implementation)"""
        logger.info(f"SMS sent to {phone_number}: {message[:50]}...")
        return True
    
    def send_slack_notification(self, channel: str, message: str) -> bool:
        """Send Slack notification (mock implementation)"""
        logger.info(f"Slack notification sent to {channel}: {message[:50]}...")
        return True
    
    def create_calendar_event(self, title: str, date: datetime.datetime, 
                            attendees: List[str]) -> str:
        """Create calendar event (mock implementation)"""
        event_id = f"EVENT-{uuid.uuid4().hex[:8]}"
        logger.info(f"Calendar event created: {event_id} - {title}")
        return event_id

class CriticalReviewAgent:
    """Agent 3 - Handles critical reviews with automation tools"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.ticketing_system = TicketingSystem()
        self.email_drafter = EmailDrafter()
        self.notification_system = NotificationSystem()
        self.processed_reviews = {}
        
        # Store management contacts
        self.store_contacts = {
            'Zudio Koramangala': {
                'manager': 'manager.koramangala@zudio.com',
                'phone': '+91-9876543210'
            },
            'Zudio Phoenix Mumbai': {
                'manager': 'manager.phoenix@zudio.com',
                'phone': '+91-9876543211'
            },
            'Zudio Jammu': {
                'manager': 'manager.jammu@zudio.com',
                'phone': '+91-9876543212'
            },
            'Zudio Bhopal': {
                'manager': 'manager.bhopal@zudio.com',
                'phone': '+91-9876543213'
            },
            'Zudio Velachery': {
                'manager': 'manager.velachery@zudio.com',
                'phone': '+91-9876543214'
            }
        }
    
    def process_critical_review(self, review: CriticalReview) -> List[ActionResult]:
        """Main function to process a critical review and execute automated actions"""
        
        if review.criticality_score <= 7:
            logger.warning(f"Review {review.review_id} has score {review.criticality_score} <= 7, skipping")
            return []
        
        logger.info(f"Processing critical review {review.review_id} with score {review.criticality_score}")
        
        # Parse and categorize the action
        action_plan = self._parse_action_plan(review.action_to_be_performed)
        
        # Execute automated actions
        results = []
        for action in action_plan:
            try:
                result = self._execute_action(review, action)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error executing action {action}: {str(e)}")
        
        # Store processing record
        self.processed_reviews[review.review_id] = {
            'review': review,
            'actions_taken': results,
            'processed_at': datetime.datetime.now()
        }
        
        return results
    
    def _parse_action_plan(self, action_description: str) -> List[ActionType]:
        """Parse the action description and map to specific action types"""
        action_mapping = {
            'ticket': ActionType.RAISE_URGENT_TICKET,
            'escalate': ActionType.ESCALATE_TO_MANAGER,
            'refund': ActionType.INITIATE_REFUND,
            'maintenance': ActionType.SCHEDULE_MAINTENANCE,
            'apology': ActionType.SEND_APOLOGY_EMAIL,
            'training': ActionType.REQUEST_STAFF_TRAINING,
            'inventory': ActionType.INVENTORY_ALERT,
            'quality': ActionType.QUALITY_CONTROL_ALERT,
            'callback': ActionType.CUSTOMER_CALLBACK,
            'visit': ActionType.STORE_VISIT_SCHEDULE,
            'social': ActionType.SOCIAL_MEDIA_RESPONSE,
            'repair': ActionType.FACILITY_REPAIR_REQUEST
        }
        
        actions = []
        action_lower = action_description.lower()
        
        for keyword, action_type in action_mapping.items():
            if keyword in action_lower:
                actions.append(action_type)
        
        # Default actions for critical reviews
        if not actions:
            actions = [ActionType.RAISE_URGENT_TICKET, ActionType.ESCALATE_TO_MANAGER]
        
        return actions
    
    def _execute_action(self, review: CriticalReview, action_type: ActionType) -> Optional[ActionResult]:
        """Execute a specific automated action"""
        
        action_id = f"ACTION-{uuid.uuid4().hex[:8]}"
        
        try:
            if action_type == ActionType.RAISE_URGENT_TICKET:
                return self._raise_urgent_ticket(review, action_id)
            
            elif action_type == ActionType.ESCALATE_TO_MANAGER:
                return self._escalate_to_manager(review, action_id)
            
            elif action_type == ActionType.SEND_APOLOGY_EMAIL:
                return self._send_apology_email(review, action_id)
            
            elif action_type == ActionType.REQUEST_STAFF_TRAINING:
                return self._request_staff_training(review, action_id)
            
            elif action_type == ActionType.SCHEDULE_MAINTENANCE:
                return self._schedule_maintenance(review, action_id)
            
            elif action_type == ActionType.CUSTOMER_CALLBACK:
                return self._schedule_customer_callback(review, action_id)
            
            elif action_type == ActionType.INVENTORY_ALERT:
                return self._send_inventory_alert(review, action_id)
            
            elif action_type == ActionType.INITIATE_REFUND:
                return self._initiate_refund_process(review, action_id)
            
            # Add more action implementations as needed
            
        except Exception as e:
            logger.error(f"Failed to execute {action_type}: {str(e)}")
            return None
    
    def _raise_urgent_ticket(self, review: CriticalReview, action_id: str) -> ActionResult:
        """Raise an urgent support ticket"""
        ticket_id = self.ticketing_system.create_ticket(
            title=f"CRITICAL: Customer Issue at {review.store_name}",
            description=f"Critical customer feedback (Score: {review.criticality_score})\n\n"
                       f"Customer: {review.customer_name}\n"
                       f"Review: {review.review_text}\n"
                       f"Recommended Action: {review.action_to_be_performed}",
            priority=Priority.CRITICAL,
            store_name=review.store_name,
            category=review.category or "general"
        )
        
        return ActionResult(
            action_id=action_id,
            action_type=ActionType.RAISE_URGENT_TICKET,
            status="completed",
            timestamp=datetime.datetime.now(),
            details={"ticket_id": ticket_id},
            follow_up_required=True
        )
    
    def _escalate_to_manager(self, review: CriticalReview, action_id: str) -> ActionResult:
        """Escalate issue to store manager"""
        store_info = self.store_contacts.get(review.store_name, {})
        
        email_context = {
            "recipient": store_info.get('manager', 'manager@zudio.com'),
            "recipient_name": "Store Manager",
            "store_name": review.store_name,
            "customer_name": review.customer_name,
            "criticality_score": review.criticality_score,
            "category": review.category or "general",
            "review_summary": review.review_text[:200] + "..." if len(review.review_text) > 200 else review.review_text,
            "action_to_be_performed": review.action_to_be_performed,
            "ticket_id": "TBD",
            "automated_actions": "Ticket raised, Manager notified"
        }
        
        email_draft = self.email_drafter.draft_email("escalation", email_context)
        
        # Send SMS alert to store manager
        if 'phone' in store_info:
            sms_message = f"URGENT: Critical customer issue at {review.store_name}. Check email immediately. Score: {review.criticality_score}"
            self.notification_system.send_sms_alert(store_info['phone'], sms_message)
        
        return ActionResult(
            action_id=action_id,
            action_type=ActionType.ESCALATE_TO_MANAGER,
            status="completed",
            timestamp=datetime.datetime.now(),
            details={
                "email_draft": email_draft,
                "manager_contact": store_info.get('manager'),
                "sms_sent": 'phone' in store_info
            },
            follow_up_required=True
        )
    
    def _send_apology_email(self, review: CriticalReview, action_id: str) -> ActionResult:
        """Draft and send apology email to customer"""
        if not review.customer_contact:
            logger.warning(f"No customer contact available for review {review.review_id}")
            return None
        
        email_context = {
            "customer_name": review.customer_name,
            "store_name": review.store_name,
            "issue_summary": review.review_text[:150] + "..." if len(review.review_text) > 150 else review.review_text,
            "ticket_id": "TBD"  # Will be filled after ticket creation
        }
        
        email_draft = self.email_drafter.draft_email("apology", email_context)
        
        return ActionResult(
            action_id=action_id,
            action_type=ActionType.SEND_APOLOGY_EMAIL,
            status="draft_created",
            timestamp=datetime.datetime.now(),
            details={"email_draft": email_draft},
            follow_up_required=True
        )
    
    def _request_staff_training(self, review: CriticalReview, action_id: str) -> ActionResult:
        """Request staff training based on feedback"""
        training_context = {
            "store_name": review.store_name,
            "category": review.category or "customer_service",
            "issue_summary": review.review_text[:200],
            "training_focus": self._determine_training_focus(review),
            "staff_list": "All customer-facing staff",
            "request_id": f"TRAIN-{action_id}",
            "priority": "high"
        }
        
        email_draft = self.email_drafter.draft_email("training_request", training_context)
        
        return ActionResult(
            action_id=action_id,
            action_type=ActionType.REQUEST_STAFF_TRAINING,
            status="request_sent",
            timestamp=datetime.datetime.now(),
            details={"email_draft": email_draft, "training_focus": training_context["training_focus"]},
            follow_up_required=True
        )
    
    def _schedule_maintenance(self, review: CriticalReview, action_id: str) -> ActionResult:
        """Schedule facility maintenance"""
        maintenance_context = {
            "store_name": review.store_name,
            "issue_type": "facility_issue",
            "issue_description": review.review_text,
            "priority": "high",
            "request_id": f"MAINT-{action_id}",
            "store_manager_contact": self.store_contacts.get(review.store_name, {}).get('manager', 'manager@zudio.com')
        }
        
        email_draft = self.email_drafter.draft_email("maintenance", maintenance_context)
        
        return ActionResult(
            action_id=action_id,
            action_type=ActionType.SCHEDULE_MAINTENANCE,
            status="scheduled",
            timestamp=datetime.datetime.now(),
            details={"email_draft": email_draft, "maintenance_type": "facility_repair"},
            follow_up_required=True
        )
    
    def _schedule_customer_callback(self, review: CriticalReview, action_id: str) -> ActionResult:
        """Schedule a customer callback"""
        callback_time = datetime.datetime.now() + datetime.timedelta(hours=4)
        
        if review.customer_contact:
            # Create calendar event for customer service team
            event_id = self.notification_system.create_calendar_event(
                title=f"Customer Callback - {review.customer_name}",
                date=callback_time,
                attendees=["customerservice@zudio.com"]
            )
        
        return ActionResult(
            action_id=action_id,
            action_type=ActionType.CUSTOMER_CALLBACK,
            status="scheduled",
            timestamp=datetime.datetime.now(),
            details={
                "callback_time": callback_time.isoformat(),
                "customer_contact": review.customer_contact,
                "event_id": event_id if review.customer_contact else None
            },
            follow_up_required=True
        )
    
    def _send_inventory_alert(self, review: CriticalReview, action_id: str) -> ActionResult:
        """Send inventory-related alert"""
        alert_message = f"Inventory issue reported at {review.store_name}: {review.review_text[:100]}"
        
        self.notification_system.send_slack_notification(
            channel="#inventory-alerts",
            message=alert_message
        )
        
        return ActionResult(
            action_id=action_id,
            action_type=ActionType.INVENTORY_ALERT,
            status="alert_sent",
            timestamp=datetime.datetime.now(),
            details={"alert_channel": "inventory-alerts", "message": alert_message},
            follow_up_required=False
        )
    
    def _initiate_refund_process(self, review: CriticalReview, action_id: str) -> ActionResult:
        """Initiate refund process"""
        # This would integrate with payment/refund system
        refund_request = {
            "customer_name": review.customer_name,
            "store_name": review.store_name,
            "reason": "Critical service issue",
            "status": "pending_approval",
            "amount": "TBD"  # Would be determined by customer service
        }
        
        return ActionResult(
            action_id=action_id,
            action_type=ActionType.INITIATE_REFUND,
            status="initiated",
            timestamp=datetime.datetime.now(),
            details=refund_request,
            follow_up_required=True
        )
    
    def _determine_training_focus(self, review: CriticalReview) -> str:
        """Determine the focus area for staff training based on review content"""
        review_lower = review.review_text.lower()
        
        if any(word in review_lower for word in ['rude', 'behavior', 'attitude', 'unprofessional']):
            return "Customer service etiquette and professionalism"
        elif any(word in review_lower for word in ['dirty', 'clean', 'hygiene', 'maintenance']):
            return "Store cleanliness and maintenance standards"
        elif any(word in review_lower for word in ['queue', 'wait', 'slow', 'billing']):
            return "Efficient checkout and queue management"
        elif any(word in review_lower for word in ['product', 'quality', 'defect', 'damaged']):
            return "Product quality control and inspection"
        else:
            return "General customer service excellence"
    
    def get_processing_summary(self, review_id: str) -> Optional[Dict]:
        """Get summary of actions taken for a specific review"""
        return self.processed_reviews.get(review_id)
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for monitoring dashboard"""
        total_processed = len(self.processed_reviews)
        
        action_counts = {}
        follow_ups_pending = 0
        
        for record in self.processed_reviews.values():
            for action in record['actions_taken']:
                action_type = action.action_type.value
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
                if action.follow_up_required:
                    follow_ups_pending += 1
        
        return {
            "total_critical_reviews_processed": total_processed,
            "actions_by_type": action_counts,
            "follow_ups_pending": follow_ups_pending,
            "tickets_created": len(self.ticketing_system.tickets),
            "processing_rate": f"{total_processed} reviews/day"  # This would be calculated properly
        }


class Sentiment(Enum):
    POSITIVE = "Positive"
    NEGATIVE = "Negative"
    CRITICAL = "Critical"
    NEUTRAL = "Neutral"

@dataclass
class Action:
    action_type: str
    status: str
    follow_up_required: bool
    timestamp: datetime.datetime = datetime.datetime.now()

# Define review structure
@dataclass
class ProcessedReview:
    text: str
    rating: Optional[float]
    sentiment: Sentiment
    summary: str
    issue: Optional[str]
    is_actionable: bool
    action: Optional[Action] = None

# --- Utility Functions ---
def fetch_gmap_place_id(place_name):
    """Fetches the Google Maps Place ID for a given place name."""
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
    """Fetches up to 5 public reviews from a Google Place ID."""
    details_url = (
        "https://maps.googleapis.com/maps/api/place/details/json"
        f"?place_id={place_id}&fields=reviews&key={api_key}"
    )
    details_resp = requests.get(details_url)
    data = details_resp.json()

    if not data.get("result") or not data["result"].get("reviews"):
        return []

    return data["result"]["reviews"]

def get_system_prompt():
    """Returns the system prompt for the GPT model."""
    return """
    You are a highly analytical and empathetic customer review processing assistant for a business.
    Your task is to analyze customer reviews, categorize their sentiment, identify the core issue if any,
    and determine if any follow-up action is required.
    
    You will be given a customer review. Your response MUST be a JSON object with the following keys:
    'sentiment': one of "Positive", "Negative", or "Critical". Critical sentiment is for reviews that are highly negative and demand immediate attention.
    'summary': a brief, one-sentence summary of the review's content.
    'issue': a brief, one-phrase description of the specific problem or issue mentioned. If there is no issue (e.g., for a positive review), state "No issue".
    'is_actionable': a boolean (true/false) indicating if the review requires a follow-up action. This is true for negative and critical reviews.
    'action_type': a string describing the type of action to take. For positive reviews, use "Thank You". For negative/critical reviews, use "Customer Outreach" or "Internal Report" or "Service Improvement".
    
    Example response for a positive review:
    {
        "sentiment": "Positive",
        "summary": "Customer praised the friendly staff and clean environment.",
        "issue": "No issue",
        "is_actionable": false,
        "action_type": "Thank You"
    }
    
    Example response for a negative review:
    {
        "sentiment": "Negative",
        "summary": "Customer complained about slow service and cold food.",
        "issue": "Slow service and cold food",
        "is_actionable": true,
        "action_type": "Customer Outreach"
    }
    
    Example response for a critical review:
    {
        "sentiment": "Critical",
        "summary": "Reviewer was extremely angry about a major hygiene violation.",
        "issue": "Hygiene violation",
        "is_actionable": true,
        "action_type": "Internal Report"
    }
    
    Ensure your entire response is a single, valid JSON object. Do not include any other text, explanations, or code outside the JSON.
    """

def process_review_with_llm(review_text: str) -> Optional[dict]:
    """Processes a single review using the LLM."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": review_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from LLM response: {e}")
        logger.error(f"Raw LLM response content: {content}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while calling the LLM: {e}")
        return None

def send_email(to_email: str, subject: str, body: str):
    """Sends a follow-up email."""
    # This is a placeholder for a real email sending service.
    # In a real app, you'd use a service like SendGrid, Mailgun, or configure an SMTP server.
    st.info(f"Simulating email send to {to_email} with subject: '{subject}'")
    st.code(body)

# --- Streamlit UI Functions ---
def setup_sidebar():
    """Sets up the Streamlit sidebar."""
    st.sidebar.image("https://placehold.co/150x150/808080/FFFFFF?text=Logo", use_column_width=True)
    st.sidebar.title("Configuration")

    with st.sidebar.expander("Google Maps & OpenAI API Keys"):
        st.info("Enter your API keys to fetch and process reviews.")
        st.session_state.gmap_api_key = st.text_input(
            "Google Maps API Key",
            type="password",
            value=st.session_state.gmap_api_key if 'gmap_api_key' in st.session_state else ''
        )
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.openai_api_key if 'openai_api_key' in st.session_state else ''
        )

    st.sidebar.markdown("---")
    st.sidebar.header("Review Processor")
    
    # Existing input and button
    st.session_state.place_name_default = st.sidebar.text_input(
        "Enter Business Name or Location",
        placeholder="e.g., 'The French Laundry' or 'Eiffel Tower Paris'"
    )
    if st.sidebar.button("Fetch and Process Reviews", use_container_width=True):
        st.session_state.place_name = st.session_state.place_name_default
        if not st.session_state.gmap_api_key or not st.session_state.openai_api_key:
            st.error("Please enter your Google Maps and OpenAI API keys first.")
        elif not st.session_state.place_name:
            st.error("Please enter a business name or location.")
        else:
            process_reviews_flow()

def process_reviews_flow():
    """The main flow for fetching and processing reviews."""
    if 'processing_done' in st.session_state and st.session_state.processing_done:
        st.session_state.processing_done = False
        st.session_state.processed_reviews = []
        st.session_state.action_results = []
    
    st.session_state.gmap_api_key = st.session_state.get('gmap_api_key', '')
    st.session_state.openai_api_key = st.session_state.get('openai_api_key', '')

    if not st.session_state.gmap_api_key or not st.session_state.openai_api_key:
        st.error("Please enter your API keys to continue.")
        return

    global client
    global api_key
    api_key = st.session_state.openai_api_key
    client = OpenAI(api_key=api_key)

    if 'place_name' not in st.session_state or not st.session_state.place_name:
        st.error("Please enter a business name or location to process.")
        return

    st.subheader(f"Processing Reviews for: **{st.session_state.place_name}**")
    
    with st.spinner("Fetching place ID..."):
        place_id = fetch_gmap_place_id(st.session_state.place_name)
        if "No place found" in place_id:
            st.error(place_id)
            return

    st.success(f"Found Place ID: {place_id}", icon="✅")

    with st.spinner("Fetching reviews..."):
        reviews_data = fetch_google_reviews(place_id, st.session_state.gmap_api_key)
        if not reviews_data:
            st.warning("No public reviews found for this place.")
            return

    st.success(f"Fetched {len(reviews_data)} reviews.", icon="✅")

    st.subheader("Processing Reviews with LLM...")
    progress_bar = st.progress(0)
    st.session_state.processed_reviews = []
    
    # Process reviews
    for i, review in enumerate(reviews_data):
        review_text = review.get('text', 'No text provided.')
        st.write(f"Processing review {i+1}/{len(reviews_data)}...")
        
        parsed_data = process_review_with_llm(review_text)
        if parsed_data:
            try:
                processed_review = ProcessedReview(
                    text=review_text,
                    rating=review.get('rating'),
                    sentiment=Sentiment(parsed_data['sentiment']),
                    summary=parsed_data['summary'],
                    issue=parsed_data['issue'],
                    is_actionable=parsed_data['is_actionable'],
                    action=Action(
                        action_type=parsed_data['action_type'],
                        status='Pending',
                        follow_up_required=parsed_data['is_actionable']
                    )
                )
                st.session_state.processed_reviews.append(processed_review)
            except (KeyError, ValueError) as e:
                st.warning(f"Skipping review {i+1} due to parsing error: {e}")
        
        progress_bar.progress((i + 1) / len(reviews_data))
        time.sleep(1) # Simulate processing time

    st.session_state.processing_done = True
    st.session_state.last_processed_place = st.session_state.place_name
    st.success("Review processing complete!")
    st.rerun()

def show_data_preview():
    """Displays a preview of the processed datasets with download buttons."""
    if 'processed_reviews' not in st.session_state or not st.session_state.processed_reviews:
        st.info("No processed data to display. Please process some reviews first.")
        return

    st.title("Data Preview")
    
    reviews_df = pd.DataFrame([
        {
            'text': r.text,
            'rating': r.rating,
            'sentiment': r.sentiment.value,
            'summary': r.summary,
            'issue': r.issue,
            'is_actionable': r.is_actionable,
            'action_type': r.action.action_type if r.action else 'N/A'
        }
        for r in st.session_state.processed_reviews
    ])

    st.subheader(f"All Reviews ({len(reviews_df)})")
    st.dataframe(reviews_df, use_container_width=True)

    # --- NEW: Download Buttons for Datasets ---
    st.header("Download Datasets")
    col1, col2, col3 = st.columns(3)

    # Positive Reviews
    positive_df = reviews_df[reviews_df['sentiment'] == Sentiment.POSITIVE.value]
    csv_positive = positive_df.to_csv(index=False).encode('utf-8')
    col1.metric("Positive Reviews", len(positive_df))
    col1.download_button(
        label="Download Positive",
        data=csv_positive,
        file_name='positive_reviews.csv',
        mime='text/csv',
        use_container_width=True
    )

    # Negative Reviews
    negative_df = reviews_df[reviews_df['sentiment'] == Sentiment.NEGATIVE.value]
    csv_negative = negative_df.to_csv(index=False).encode('utf-8')
    col2.metric("Negative Reviews", len(negative_df))
    col2.download_button(
        label="Download Negative",
        data=csv_negative,
        file_name='negative_reviews.csv',
        mime='text/csv',
        use_container_width=True
    )

    # Critical Reviews
    critical_df = reviews_df[reviews_df['sentiment'] == Sentiment.CRITICAL.value]
    csv_critical = critical_df.to_csv(index=False).encode('utf-8')
    col3.metric("Critical Reviews", len(critical_df))
    col3.download_button(
        label="Download Critical",
        data=csv_critical,
        file_name='critical_reviews.csv',
        mime='text/csv',
        use_container_width=True
    )

    st.markdown("---")
    
    st.subheader("Positive Reviews")
    st.dataframe(positive_df, use_container_width=True)
    
    st.subheader("Negative Reviews")
    st.dataframe(negative_df, use_container_width=True)

    st.subheader("Critical Reviews")
    st.dataframe(critical_df, use_container_width=True)

def create_wordcloud(text_data: str, title: str):
    """Generates and displays a word cloud from text data."""
    if not text_data.strip():
        st.info(f"No text data to generate word cloud for {title}.")
        return

    # Generate a word cloud image
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color="white",
        colormap='viridis'
    ).generate(text_data)

    # Display the generated image
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=20)
    ax.axis('off')
    st.pyplot(fig)

def show_dashboard():
    """Displays the analytical dashboard."""
    if 'processed_reviews' not in st.session_state or not st.session_state.processed_reviews:
        st.info("No processed data to build the dashboard. Please process some reviews first.")
        return

    st.title("Reviews Dashboard")
    st.subheader(f"Analytics for {st.session_state.last_processed_place}")
    
    reviews_df = pd.DataFrame([
        {
            'text': r.text,
            'rating': r.rating,
            'sentiment': r.sentiment.value,
            'summary': r.summary,
            'issue': r.issue,
            'is_actionable': r.is_actionable,
            'action_type': r.action.action_type if r.action else 'N/A'
        }
        for r in st.session_state.processed_reviews
    ])
    
    # Overall Sentiment Distribution
    st.subheader("Overall Sentiment Distribution")
    sentiment_counts = reviews_df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    fig = px.pie(
        sentiment_counts,
        values='count',
        names='sentiment',
        title='Distribution of Review Sentiments',
        hole=0.3,
        color='sentiment',
        color_discrete_map={
            'Positive': 'green',
            'Negative': 'orange',
            'Critical': 'red',
            'Neutral': 'gray'
        }
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # --- NEW: Comparative Widgets ---
    st.header("Comparative Analysis")
    
    # Bar Chart: Average Rating per Sentiment
    st.subheader("Average Rating per Sentiment")
    avg_ratings = reviews_df.groupby('sentiment')['rating'].mean().reset_index()
    fig = px.bar(
        avg_ratings, 
        x='sentiment', 
        y='rating', 
        color='sentiment',
        title='Average Rating by Sentiment Category',
        labels={'rating': 'Average Rating', 'sentiment': 'Sentiment'},
        color_discrete_map={
            'Positive': 'green',
            'Negative': 'orange',
            'Critical': 'red',
            'Neutral': 'gray'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    # Bar Chart: Actionability by Sentiment
    st.subheader("Actionability by Sentiment")
    actionable_counts = reviews_df.groupby('sentiment')['is_actionable'].sum().reset_index()
    actionable_counts.columns = ['sentiment', 'actionable_count']
    fig = px.bar(
        actionable_counts, 
        x='sentiment', 
        y='actionable_count', 
        color='sentiment',
        title='Number of Actionable Reviews by Sentiment',
        labels={'actionable_count': 'Count', 'sentiment': 'Sentiment'},
        color_discrete_map={
            'Positive': 'green',
            'Negative': 'orange',
            'Critical': 'red',
            'Neutral': 'gray'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- NEW: Word Clouds ---
    st.header("Word Clouds of Key Issues")

    # Word Cloud for Positive Reviews
    positive_reviews_text = " ".join(
        reviews_df[reviews_df['sentiment'] == 'Positive']['text'].dropna()
    )
    create_wordcloud(positive_reviews_text, "Positive Reviews Word Cloud")
    
    # Word Cloud for Negative Reviews
    negative_reviews_text = " ".join(
        reviews_df[reviews_df['sentiment'].isin(['Negative', 'Critical'])]['issue'].dropna()
    )
    create_wordcloud(negative_reviews_text, "Negative & Critical Issues Word Cloud")
    
    st.subheader("Reviews by Rating")
    fig = px.histogram(
        reviews_df,
        x='rating',
        color='sentiment',
        title='Distribution of Ratings by Sentiment',
        nbins=5,
        labels={'rating': 'Rating', 'count': 'Number of Reviews'},
        category_orders={'rating': sorted(reviews_df['rating'].unique())},
        color_discrete_map={
            'Positive': 'green',
            'Negative': 'orange',
            'Critical': 'red',
            'Neutral': 'gray'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

def show_actions():
    """Displays the action log and follow-up options."""
    st.title("Actions Log")
    if 'processed_reviews' in st.session_state and st.session_state.processed_reviews:
        action_results = []
        for r in st.session_state.processed_reviews:
            if r.is_actionable and r.action:
                action_results.append({
                    'Review Text': r.text,
                    'Action Type': r.action.action_type,
                    'Follow-up Required': r.action.follow_up_required,
                    'Status': r.action.status,
                    'Timestamp': r.action.timestamp
                })
        
        if action_results:
            df_actions = pd.DataFrame(action_results)
            st.dataframe(df_actions, use_container_width=True)
            
            # Follow-up actions
            st.subheader("Follow-up Actions")
            actionable_reviews = [r for r in st.session_state.processed_reviews if r.is_actionable]
            
            for i, review in enumerate(actionable_reviews):
                with st.expander(f"Review from {review.sentiment.value} category"):
                    st.write(f"**Summary:** {review.summary}")
                    st.write(f"**Original Text:** {review.text}")
                    
                    if st.button(f"Send Follow-up Email (ID: {i+1})"):
                        subject = f"Follow-up regarding your recent review"
                        body = f"Dear Customer,\n\nThank you for your feedback regarding '{review.issue}'. We apologize for the inconvenience and would like to learn more about your experience.\n\nSincerely,\nThe Team"
                        send_email("customer@example.com", subject, body)
                        st.session_state.action_results.append(Action(
                            action_type="Email Sent",
                            status="Completed",
                            follow_up_required=False,
                            timestamp=datetime.datetime.now()
                        ))
        else:
            st.info("No actionable reviews found.")
    else:
        st.info("No reviews have been processed yet.")

def show_settings():
    """Displays a settings page."""
    st.title("App Settings")
    st.info("This is a placeholder for future settings.")

# --- Main App Logic ---
def main():
    st.set_page_config(
        page_title="Customer Review Processor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if 'page' not in st.session_state:
        st.session_state.page = 'dashboard'
    if 'processed_reviews' not in st.session_state:
        st.session_state.processed_reviews = []
    if 'action_results' not in st.session_state:
        st.session_state.action_results = []
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False
    if 'gmap_api_key' not in st.session_state:
        st.session_state.gmap_api_key = ''
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ''
    if 'last_processed_place' not in st.session_state:
        st.session_state.last_processed_place = 'N/A'
    
    setup_sidebar()

    st.sidebar.markdown("---")
    
    # --- New Section for Custom Store Processing ---
    st.sidebar.header("Test Custom Store")
    st.session_state.place_name = st.sidebar.text_input(
        "Enter Store Name to Test",
        placeholder="e.g., 'Starbucks Times Square'",
        key="custom_store_name_input"
    )
    if st.sidebar.button("Process Custom Store", use_container_width=True):
        if not st.session_state.gmap_api_key or not st.session_state.openai_api_key:
            st.error("Please enter your API keys first.")
        elif not st.session_state.place_name:
            st.error("Please enter a custom store name.")
        else:
            process_reviews_flow()

    st.sidebar.markdown("---")
    st.sidebar.header("Navigation")
    if st.sidebar.button("Dashboard", use_container_width=True):
        st.session_state.page = 'dashboard'
    if st.sidebar.button("Data Preview", use_container_width=True):
        st.session_state.page = 'data'
    if st.sidebar.button("Actions Log", use_container_width=True):
        st.session_state.page = 'actions'

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Customer Review Processor v2.0**")
    st.sidebar.markdown("Built with Streamlit & OpenAI")

    if st.session_state.page == 'data':
        show_data_preview()
    elif st.session_state.page == 'dashboard':
        show_dashboard()
    elif st.session_state.page == 'actions':
        show_actions()

if __name__ == "__main__":
    main()
