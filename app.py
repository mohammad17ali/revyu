import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from collections import Counter
import json
import time
import io
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
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

from functions import fetch_gmap_place_id, fetch_google_reviews, process_store_list, FashionFeedbackProcessor, CriticalReviewAgent, create_download_button, CriticalReview
from data import get_stores_by_type, get_all_store_names, get_store_details

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client and other variables
client = None
api_key = None
processor = None

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

st.set_page_config(
    page_title="Revvyu",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'positive_dataset' not in st.session_state:
    st.session_state.positive_dataset = None
if 'negative_dataset' not in st.session_state:
    st.session_state.negative_dataset = None
if 'critical_dataset' not in st.session_state:
    st.session_state.critical_dataset = None
if 'action_results' not in st.session_state:
    st.session_state.action_results = []
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""

# Sidebar for navigation
st.sidebar.title("🛍️ Revvyu - Feedback Processor")

# Initialize page in session state
if 'page' not in st.session_state:
    st.session_state.page = "Review Processor"

page = st.sidebar.selectbox("Select Page", ["Review Processor", "Dashboard", "Action Center"], index=["Review Processor", "Dashboard", "Action Center"].index(st.session_state.page))

# Update session state if page changed via sidebar
if page != st.session_state.page:
    st.session_state.page = page

# API Key input in sidebar
with st.sidebar.expander("API Configuration"):
    # Set default values if not already set
    default_openai_key = st.session_state.openai_api_key if st.session_state.openai_api_key else "sk-proj-cXtRTFMtcL-FkSQ1ZqnwcYUva2AtRmNDJY0FiRkyD7ORo5EGWPALxmRBnZq9FQOhZz7HlBcq_pT3BlbkFJuYXMLs3p6-EiBW0edIQOX9TD2ddylWxoo5BFLfmMwlr8Ypg7ZUDDLQ6hJxCtC27Xa7eKQlnHY"
    default_gmaps_key = "IzaSyBZnZH9E03uTaEZSvD002FCWbkx9iguTXM"
    
    openai_api_key = st.text_input("OpenAI API Key", type="password", value=default_openai_key)
    google_maps_api_key = st.text_input("Google Maps API Key", type="password", key="gmaps_key", value=default_gmaps_key)
    
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        client = OpenAI(api_key=openai_api_key)
        st.success("OpenAI API Key configured!")
    
    if google_maps_api_key:
        api_key = google_maps_api_key
        st.success("Google Maps API Key configured!")
    # Initialize global variables
if st.session_state.openai_api_key:
    client = OpenAI(api_key=st.session_state.openai_api_key)
    processor = FashionFeedbackProcessor(st.session_state.openai_api_key)


# Helper functions
def process_single_review(review_text, store_name, api_key):
    """Process a single review through the pipeline"""
    try:
        processor = FashionFeedbackProcessor(api_key)
        # This would need to be adapted based on your actual single review processing method
        result = processor.process_single_review(review_text)
        return result
    except Exception as e:
        st.error(f"Error processing review: {str(e)}")
        return None

def calculate_satisfaction_scores(positive_dataset):
    """Calculate average satisfaction scores by store"""
    if not positive_dataset:
        return {}
    
    store_scores = {}
    for review in positive_dataset:
        store = review.get('store_branch', 'Unknown')
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
        store = review.get('store_branch', 'Unknown')
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
        if review.get('store_branch') in selected_stores:
            issue = review.get('specific_issue')
            if issue:
                issues.append(issue)
    
    issue_counts = Counter(issues)
    return issue_counts.most_common(5)

def extract_positive_words(positive_dataset, selected_stores):
    """Extract individual words from positive praise quotes for wordcloud"""
    if not positive_dataset or not selected_stores:
        return ""
    
    all_words = []
    for review in positive_dataset:
        if review.get('store_branch') in selected_stores:
            # Extract words from customer_praise_quotes
            quotes = review.get('customer_praise_quotes', [])
            for quote in quotes:
                if quote:
                    # Split into words and clean them
                    words = quote.lower().split()
                    for word in words:
                        # Remove punctuation and keep only alphabetic words
                        clean_word = ''.join(char for char in word if char.isalpha())
                        if len(clean_word) > 2:  # Only keep words longer than 2 characters
                            all_words.append(clean_word)
    
    return ' '.join(all_words)

def get_city_wise_analysis(positive_dataset, negative_dataset, selected_stores):
    """Get city-wise analysis for satisfaction and criticality scores"""
    if not selected_stores:
        return {}, {}
    
    # Get store details to map store names to cities
    from data import get_store_details
    
    # Filter datasets for selected stores only
    selected_positive = [r for r in (positive_dataset or []) if r.get('store_branch') in selected_stores]
    selected_negative = [r for r in (negative_dataset or []) if r.get('store_branch') in selected_stores]
    
    # Group by city
    city_satisfaction = {}
    city_criticality = {}
    
    for review in selected_positive:
        store_name = review.get('store_branch', 'Unknown')
        store_details = get_store_details(store_name)
        if store_details:
            city = store_details.get('City', 'Unknown')
            score = review.get('satisfaction_score', 0)
            
            if city not in city_satisfaction:
                city_satisfaction[city] = {'scores': [], 'stores': set()}
            city_satisfaction[city]['scores'].append(score)
            city_satisfaction[city]['stores'].add(store_name)
    
    for review in selected_negative:
        store_name = review.get('store_branch', 'Unknown')
        store_details = get_store_details(store_name)
        if store_details:
            city = store_details.get('City', 'Unknown')
            score = review.get('criticality_score', 0)
            
            if city not in city_criticality:
                city_criticality[city] = {'scores': [], 'stores': set()}
            city_criticality[city]['scores'].append(score)
            city_criticality[city]['stores'].add(store_name)
    
    # Calculate averages and convert sets to lists
    for city in city_satisfaction:
        scores = city_satisfaction[city]['scores']
        city_satisfaction[city] = {
            'avg_score': sum(scores) / len(scores) if scores else 0,
            'stores': list(city_satisfaction[city]['stores']),
            'review_count': len(scores)
        }
    
    for city in city_criticality:
        scores = city_criticality[city]['scores']
        city_criticality[city] = {
            'avg_score': sum(scores) / len(scores) if scores else 0,
            'stores': list(city_criticality[city]['stores']),
            'review_count': len(scores)
        }
    
    return city_satisfaction, city_criticality

# PAGE 1: REVIEW PROCESSOR
if st.session_state.page == "Review Processor":
    st.title("🔄 Review Processor Pipeline")
    st.markdown("Process customer reviews through our AI-powered analysis pipeline")
    
    if not st.session_state.openai_api_key:
        st.warning("Please configure your OpenAI API key in the sidebar to proceed.")
        st.stop()
    
    tab1, tab2 = st.tabs(["🏪 Multiple Store Processing", "📝 Single Review Testing"])
    
    with tab1:
        st.header("Bulk Store Processing")
        st.markdown("Select multiple stores to process all their reviews through the pipeline")
        
        # Store type toggle
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            store_type = st.radio(
                "Select Store Type:",
                ["Westside", "Zudio"],
                horizontal=True,
                key="store_type_toggle"
            )
        
        # Get stores based on selected type
        if store_type == "Westside":
            available_stores = get_stores_by_type("westside")
        else:
            available_stores = get_stores_by_type("zudio")
        
        # Store selection
        selected_stores = st.multiselect(
            f"Select {store_type} stores to process:",
            available_stores,
            default=available_stores[:2] if len(available_stores) >= 2 else available_stores
        )
        
        col1, col2 = st.columns(2)
        with col1:
            process_button = st.button("🚀 Process Selected Stores", type="primary")
        with col2:
            if st.session_state.processed_data:
                st.success(f"Last processed: {len(selected_stores)} stores")

        
        if process_button and selected_stores:
            with st.spinner("Processing reviews... This may take a few minutes"):
                try:
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process stores
                    status_text.text("Initializing processor...")
                    progress_bar.progress(10)
                    
                    status_text.text("Processing store reviews...")
                    progress_bar.progress(30)
                    processor = FashionFeedbackProcessor(st.session_state.openai_api_key)
                    
                    positive_dataset, negative_dataset, critical_dataset = process_store_list(selected_stores, api_key, processor)
                    progress_bar.progress(70)
                    
                    # Store in session state
                    st.session_state.positive_dataset = positive_dataset
                    st.session_state.negative_dataset = negative_dataset
                    st.session_state.critical_dataset = critical_dataset
                    st.session_state.processed_data = True
                    
                    progress_bar.progress(100)
                    status_text.text("Processing complete!")
                    
                    # Display summary
                    st.success("✅ Processing completed successfully!")
                    
                    # Display metrics immediately after processing
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Positive Reviews", len(positive_dataset))
                    with col2:
                        st.metric("Negative Reviews", len(negative_dataset))
                    with col3:
                        st.metric("Critical Reviews", len(critical_dataset))
                    
                    # Display download buttons immediately after processing
                    st.divider()
                    st.subheader("📥 Download Datasets")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        create_download_button(
                            positive_dataset,
                            "positive_reviews.csv",
                            "📊 Download Positive Reviews",
                            key="download_positive_immediate"
                        )
                    with col2:
                        create_download_button(
                            negative_dataset,
                            "negative_reviews.csv",
                            "📊 Download Negative Reviews",
                            key="download_negative_immediate"
                        )
                    with col3:
                        create_download_button(
                            critical_dataset,
                            "critical_reviews.csv",
                            "📊 Download Critical Reviews",
                            key="download_critical_immediate"
                        )
                    
                    # Navigation buttons after immediate processing
                    st.divider()
                    st.subheader("🎯 Next Steps")
                    st.markdown("Choose your next action based on the processed datasets:")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📊 View Insights", type="primary", use_container_width=True, help="Navigate to the Dashboard to view analytics and insights", key="insights_immediate"):
                            st.session_state.page = "Dashboard"
                            st.rerun()
                    
                    with col2:
                        if st.button("⚡ Take Actions", type="secondary", use_container_width=True, help="Navigate to the Action Center to process critical reviews", key="actions_immediate"):
                            st.session_state.page = "Action Center"
                            st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
        
        # Show download buttons and metrics if data exists (for when user returns to page)
        if st.session_state.processed_data and (st.session_state.positive_dataset or st.session_state.negative_dataset or st.session_state.critical_dataset):
            st.divider()
            st.subheader("📊 Processing Results")
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positive Reviews", len(st.session_state.positive_dataset or []))
            with col2:
                st.metric("Negative Reviews", len(st.session_state.negative_dataset or []))
            with col3:
                st.metric("Critical Reviews", len(st.session_state.critical_dataset or []))
            
            # Display download buttons
            st.subheader("📥 Download Datasets")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                create_download_button(
                    st.session_state.positive_dataset,
                    "positive_reviews.csv",
                    "📊 Download Positive Reviews",
                    key="download_positive_persistent"
                )
            with col2:
                create_download_button(
                    st.session_state.negative_dataset,
                    "negative_reviews.csv",
                    "📊 Download Negative Reviews",
                    key="download_negative_persistent"
                )
            with col3:
                create_download_button(
                    st.session_state.critical_dataset,
                    "critical_reviews.csv",
                    "📊 Download Critical Reviews",
                    key="download_critical_persistent"
                )
            
            # Navigation buttons
            st.divider()
            st.subheader("🎯 Next Steps")
            st.markdown("Choose your next action based on the processed datasets:")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 View Insights", type="primary", use_container_width=True, help="Navigate to the Dashboard to view analytics and insights"):
                    st.session_state.page = "Dashboard"
                    st.rerun()
            
            with col2:
                if st.button("⚡ Take Actions", type="secondary", use_container_width=True, help="Navigate to the Action Center to process critical reviews"):
                    st.session_state.page = "Action Center"
                    st.rerun()
    
    with tab2:
        st.header("Single Review Testing")
        st.markdown("Test the pipeline with a single review to see how it works")
        
        # Review type hint buttons
        st.subheader("📝 Review Type Hints")
        st.markdown("Click a button below to fill the text area with a sample review of that type:")
        
        col1, col2, col3 = st.columns(3)
        
        # Sample reviews based on type
        sample_reviews = {
            "Positive": "Amazing experience at this store! The staff was so helpful and friendly. Found exactly what I was looking for. The kurta collection is fantastic and very reasonably priced. Will definitely come back and recommend to friends!",
            "Negative": "Very disappointed with the service. The staff was rude and unhelpful. The fitting rooms were dirty and the clothes quality was poor. Waited for 30 minutes just to try on a dress. Not coming back here again.",
            "Critical": "Terrible experience! The store manager was extremely rude and refused to help with a return. The product was defective but they wouldn't acknowledge it. This is completely unacceptable. I'm posting this on social media to warn others."
        }
        
        with col1:
            if st.button("😊 Positive Review", help="Click to load a positive review example"):
                st.session_state.sample_review = sample_reviews["Positive"]
        
        with col2:
            if st.button("😞 Negative Review", help="Click to load a negative review example"):
                st.session_state.sample_review = sample_reviews["Negative"]
        
        with col3:
            if st.button("😡 Critical Review", help="Click to load a critical review example"):
                st.session_state.sample_review = sample_reviews["Critical"]
        
        # Initialize session state for sample review
        if 'sample_review' not in st.session_state:
            st.session_state.sample_review = sample_reviews["Positive"]
        
        st.divider()
        
        # Review text input with highlighting
        st.subheader("📝 Enter Review Text")
        review_text = st.text_area(
            "Review Text:",
            value=st.session_state.sample_review,
            height=150,
            help="Enter a customer review to process through the pipeline",
            key="review_text_area"
        )
        
        # Add some visual highlighting with custom CSS
        st.markdown("""
        <style>
        .stTextArea > div > div > textarea {
            border: 2px solid #1f77b4 !important;
            border-radius: 8px !important;
            box-shadow: 0 0 10px rgba(31, 119, 180, 0.3) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze Review", type="primary", use_container_width=True)
        
        if analyze_button:
            if review_text:
                with st.spinner("Analyzing review through the pipeline..."):
                    try:
                        # Process single review through the pipeline
                        processor = FashionFeedbackProcessor(st.session_state.openai_api_key)
                        result = processor.process_single_review(review_text)
                        
                        if result:
                            st.success("✅ Review processed successfully!")
                            
                            # Display results section
                            st.divider()
                            st.subheader("📊 Analysis Results")
                            
                            # Show classification with appropriate styling
                            sentiment = result.get('sentiment', 'Unknown')
                            if sentiment == 'Positive':
                                st.success(f"🎉 Classification: {sentiment}")
                                score = result.get('satisfaction_score', 0)
                                st.metric("Satisfaction Score", f"{score}/10")
                                
                                # Show positive insights
                                if 'positive_aspects' in result:
                                    st.info(f"**Key Positive Aspects:** {result['positive_aspects']}")
                                    
                            elif sentiment == 'Negative':
                                st.warning(f"⚠️ Classification: {sentiment}")
                                score = result.get('criticality_score', 0)
                                st.metric("Criticality Score", f"{score}/10")
                                
                                # Show specific issues
                                if 'specific_issue' in result:
                                    st.error(f"**Specific Issue:** {result['specific_issue']}")
                                
                                if score > 7:
                                    st.error("🚨 This review requires immediate attention!")
                                    
                            elif sentiment == 'Critical':
                                st.error(f"🚨 Classification: {sentiment}")
                                score = result.get('criticality_score', 0)
                                st.metric("Criticality Score", f"{score}/10")
                                
                                # Show critical issues
                                if 'specific_issue' in result:
                                    st.error(f"**Critical Issue:** {result['specific_issue']}")
                                
                                if 'action_to_be_performed' in result:
                                    st.error(f"**Recommended Action:** {result['action_to_be_performed']}")
                            
                            # Display full results in expandable section
                            with st.expander("📋 Complete Analysis Details", expanded=False):
                                st.json(result)
                    
                    except Exception as e:
                        st.error(f"Error processing review: {str(e)}")
                        st.error("Please check your OpenAI API key and try again.")
            else:
                st.warning("Please enter a review text before analyzing.")

# PAGE 2: DASHBOARD
elif st.session_state.page == "Dashboard":
    st.title("📊 Analytics Dashboard")
    
    if not st.session_state.processed_data:
        st.warning("Please process some reviews first using the Review Processor page.")
        st.stop()
    
    st.markdown("Comprehensive analytics from processed customer reviews")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Positive", len(st.session_state.positive_dataset or []))
    with col2:
        st.metric("Total Negative", len(st.session_state.negative_dataset or []))
    with col3:
        st.metric("Critical Issues", len(st.session_state.critical_dataset or []))
    with col4:
        critical_high = len([r for r in (st.session_state.critical_dataset or []) if r.get('criticality_score', 0) > 7])
        st.metric("High Priority", critical_high)
    
    st.divider()
    
    # Store Leaderboards
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top Performing Stores")
        st.markdown("*Based on customer satisfaction scores*")
        
        satisfaction_scores = calculate_satisfaction_scores(st.session_state.positive_dataset)
        if satisfaction_scores:
            # Sort by score
            sorted_stores = sorted(satisfaction_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Create bar chart
            stores = [item[0] for item in sorted_stores]
            scores = [item[1] for item in sorted_stores]
            
            fig = px.bar(
                x=scores, 
                y=stores, 
                orientation='h',
                title="Average Satisfaction Score by Store",
                color=scores,
                color_continuous_scale="Greens"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="satisfaction_chart")
            
            # Show top 3
            st.markdown("**Top 3 Performers:**")
            for i, (store, score) in enumerate(sorted_stores[:3]):
                st.write(f"{i+1}. {store}: {score:.1f}/10")
    
    with col2:
        st.subheader("⚠️ Stores Needing Attention")
        st.markdown("*Based on negative feedback intensity*")
        
        dissatisfaction_scores = calculate_dissatisfaction_scores(st.session_state.negative_dataset)
        if dissatisfaction_scores:
            # Sort by score (higher is worse)
            sorted_stores = sorted(dissatisfaction_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Create bar chart
            stores = [item[0] for item in sorted_stores]
            scores = [item[1] for item in sorted_stores]
            
            fig = px.bar(
                x=scores, 
                y=stores, 
                orientation='h',
                title="Average Criticality Score by Store",
                color=scores,
                color_continuous_scale="Reds"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="criticality_chart")
            
            # Show top 3 issues
            st.markdown("**Top 3 Areas for Improvement:**")
            for i, (store, score) in enumerate(sorted_stores[:3]):
                st.write(f"{i+1}. {store}: {score:.1f}/10")
    
    st.divider()
    
    # Common Issues Analysis
    st.subheader("🔍 Common Issues Analysis")
    st.markdown("Select stores to identify the most common customer complaints")
    
    # Store selector for issues analysis
    all_stores = list(set([r.get('store_branch', 'Unknown') for r in (st.session_state.negative_dataset or [])]))
    selected_stores_issues = st.multiselect(
        "Select stores to analyze:",
        all_stores,
        default=all_stores[:3] if len(all_stores) >= 3 else all_stores
    )
    
    if selected_stores_issues:
        common_issues = get_common_issues(st.session_state.negative_dataset, selected_stores_issues)
        
        if common_issues:
            # Create horizontal bar chart
            issues = [issue[0] for issue in common_issues]
            counts = [issue[1] for issue in common_issues]
            
            fig = px.bar(
                x=counts,
                y=issues,
                orientation='h',
                title=f"Top 5 Common Issues in Selected Stores",
                color=counts,
                color_continuous_scale="Oranges"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, key="issues_chart")
            
            # Display in table format
            st.markdown("**Issue Breakdown:**")
            df_issues = pd.DataFrame(common_issues, columns=['Issue', 'Frequency'])
            st.dataframe(df_issues, use_container_width=True)
    
    # Additional Analytics
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sentiment Distribution")
        if st.session_state.processed_data:
            sentiment_data = {
                'Positive': len(st.session_state.positive_dataset or []),
                'Negative': len(st.session_state.negative_dataset or []),
                'Critical': len([r for r in (st.session_state.critical_dataset or []) if r.get('criticality_score', 0) > 7])
            }
            
            fig = px.pie(
                values=list(sentiment_data.values()),
                names=list(sentiment_data.keys()),
                color_discrete_sequence=['#00cc44', '#ff9900', '#ff3333']
            )
            st.plotly_chart(fig, use_container_width=True, key="sentiment_pie_2")
    
    with col2:
        st.subheader("⏰ Processing Summary")
        processing_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.info(f"Last updated: {processing_time}")
        
        # Summary stats
        if st.session_state.negative_dataset:
            avg_criticality = sum([r.get('criticality_score', 0) for r in st.session_state.negative_dataset]) / len(st.session_state.negative_dataset)
            st.metric("Avg Criticality", f"{avg_criticality:.1f}/10")
        
        if st.session_state.positive_dataset:
            avg_satisfaction = sum([r.get('satisfaction_score', 0) for r in st.session_state.positive_dataset]) / len(st.session_state.positive_dataset)
            st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/10")
    
    # Additional Analytics
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sentiment Distribution")
        if st.session_state.processed_data:
            sentiment_data = {
                'Positive': len(st.session_state.positive_dataset or []),
                'Negative': len(st.session_state.negative_dataset or []),
                'Critical': len([r for r in (st.session_state.critical_dataset or []) if r.get('criticality_score', 0) > 7])
            }
            
            fig = px.pie(
                values=list(sentiment_data.values()),
                names=list(sentiment_data.keys()),
                color_discrete_sequence=['#00cc44', '#ff9900', '#ff3333']
            )
            st.plotly_chart(fig, use_container_width=True, key="sentiment_pie_3")
    
    with col2:
        st.subheader("⏰ Processing Summary")
        processing_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.info(f"Last updated: {processing_time}")
        
        # Summary stats
        if st.session_state.negative_dataset:
            avg_criticality = sum([r.get('criticality_score', 0) for r in st.session_state.negative_dataset]) / len(st.session_state.negative_dataset)
            st.metric("Avg Criticality", f"{avg_criticality:.1f}/10")
        
        if st.session_state.positive_dataset:
            avg_satisfaction = sum([r.get('satisfaction_score', 0) for r in st.session_state.positive_dataset]) / len(st.session_state.positive_dataset)
            st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/10")
    
    # New Analytics Section
    st.divider()
    st.subheader("🔍 Advanced Analytics")
    
    # Store selector for advanced analytics
    all_stores_advanced = list(set([r.get('store_branch', 'Unknown') for r in (st.session_state.positive_dataset or []) + (st.session_state.negative_dataset or [])]))
    selected_stores_advanced = st.multiselect(
        "Select stores for advanced analytics:",
        all_stores_advanced,
        default=all_stores_advanced[:3] if len(all_stores_advanced) >= 3 else all_stores_advanced,
        key="advanced_analytics_stores"
    )
    
    if selected_stores_advanced:
        # Word Cloud Widget
        st.subheader("☁️ Positive Words Word Cloud")
        st.markdown("*Most frequently used positive words from customer praise quotes*")
        
        positive_words_text = extract_positive_words(st.session_state.positive_dataset, selected_stores_advanced)
        
        if positive_words_text:
            # Generate word cloud
            try:
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap='viridis',
                    max_words=50,
                    relative_scaling=0.5
                ).generate(positive_words_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.set_title("Positive Words from Customer Reviews", fontsize=16)
                ax.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating word cloud: {str(e)}")
        else:
            st.info("No positive words found for the selected stores.")
        
        # City-wise Analysis
        st.subheader("🏙️ City-wise Performance Analysis")
        st.markdown("*Average satisfaction and criticality scores by city for selected stores*")
        
        city_satisfaction, city_criticality = get_city_wise_analysis(
            st.session_state.positive_dataset, 
            st.session_state.negative_dataset, 
            selected_stores_advanced
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 City-wise Satisfaction Scores")
            if city_satisfaction:
                # Sort cities by average satisfaction score
                sorted_cities = sorted(city_satisfaction.items(), key=lambda x: x[1]['avg_score'], reverse=True)
                
                # Create data for table
                city_data = []
                for city, data in sorted_cities:
                    city_data.append({
                        'City': city,
                        'Avg Satisfaction Score': f"{data['avg_score']:.1f}/10",
                        'Review Count': data['review_count'],
                        'Stores': ', '.join(data['stores'])
                    })
                
                df_city_satisfaction = pd.DataFrame(city_data)
                st.dataframe(df_city_satisfaction, use_container_width=True)
                
                # Create bar chart
                cities = [item[0] for item in sorted_cities]
                scores = [item[1]['avg_score'] for item in sorted_cities]
                
                fig = px.bar(
                    x=cities,
                    y=scores,
                    title="Average Satisfaction Score by City",
                    color=scores,
                    color_continuous_scale="Greens"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True, key="city_satisfaction_chart")
            else:
                st.info("No satisfaction data available for the selected stores.")
        
        with col2:
            st.subheader("⚠️ City-wise Criticality Scores")
            if city_criticality:
                # Sort cities by average criticality score (higher is worse)
                sorted_cities = sorted(city_criticality.items(), key=lambda x: x[1]['avg_score'], reverse=True)
                
                # Create data for table
                city_data = []
                for city, data in sorted_cities:
                    city_data.append({
                        'City': city,
                        'Avg Criticality Score': f"{data['avg_score']:.1f}/10",
                        'Review Count': data['review_count'],
                        'Stores': ', '.join(data['stores'])
                    })
                
                df_city_criticality = pd.DataFrame(city_data)
                st.dataframe(df_city_criticality, use_container_width=True)
                
                # Create bar chart
                cities = [item[0] for item in sorted_cities]
                scores = [item[1]['avg_score'] for item in sorted_cities]
                
                fig = px.bar(
                    x=cities,
                    y=scores,
                    title="Average Criticality Score by City",
                    color=scores,
                    color_continuous_scale="Reds"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True, key="city_criticality_chart")
            else:
                st.info("No criticality data available for the selected stores.")
        
        # City Analysis Summary
        if city_satisfaction or city_criticality:
            st.subheader("📋 City Analysis Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if city_satisfaction:
                    best_city = max(city_satisfaction.items(), key=lambda x: x[1]['avg_score'])
                    st.success(f"🏆 **Best Performing City:** {best_city[0]} (Score: {best_city[1]['avg_score']:.1f}/10)")
                    st.write(f"Stores: {', '.join(best_city[1]['stores'])}")
            
            with col2:
                if city_criticality:
                    worst_city = max(city_criticality.items(), key=lambda x: x[1]['avg_score'])
                    st.error(f"⚠️ **City Needing Attention:** {worst_city[0]} (Score: {worst_city[1]['avg_score']:.1f}/10)")
                    st.write(f"Stores: {', '.join(worst_city[1]['stores'])}")
    else:
        st.info("Please select stores to view advanced analytics.")

# PAGE 3: ACTION CENTER
elif st.session_state.page == "Action Center":
    st.title("⚡ Action Center")
    st.markdown("Review and execute actions for critical customer feedback")
    
    if not st.session_state.processed_data:
        st.warning("Please process some reviews first using the Review Processor page.")
        st.stop()
    
    # Initialize session state
    if 'selected_review' not in st.session_state:
        st.session_state.selected_review = None
    if 'executing_actions' not in st.session_state:
        st.session_state.executing_actions = False
    if 'action_log' not in st.session_state:
        st.session_state.action_log = []
    
    # Get critical reviews
    critical_reviews = [r for r in (st.session_state.critical_dataset or []) 
                       if r.get('criticality_score', 0) > 7 and r.get('status') != 'processed']
    
    if not critical_reviews:
        st.info("No critical reviews requiring immediate action found.")
    else:
        st.warning(f"Found {len(critical_reviews)} critical reviews requiring action")
        
        # Create two-column layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📋 Critical Reviews")
            st.markdown("*Click on a review to view details and take action*")
            
            # Display list of critical reviews
            for i, review in enumerate(critical_reviews):
                # Get store name or show "Unknown" if null
                store_name = review.get('store_branch') or review.get('store_name') or "Unknown Store"
                
                # Truncate review text to first 50 words
                review_text = review.get('original_review', '')
                words = review_text.split()[:50]
                truncated_text = ' '.join(words)
                if len(review_text.split()) > 50:
                    truncated_text += "..."
                
                # Get action to be performed
                action = review.get('action_to_be_performed', 'No action specified')
                
                # Create a card-like display for each review
                with st.container():
                    # Highlight the selected review
                    is_selected = st.session_state.selected_review == i
                    if is_selected:
                        st.markdown("""
                        <div style="border: 2px solid #1f77b4; border-radius: 8px; padding: 10px; margin: 5px 0; background-color: #f0f8ff;">
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; margin: 5px 0; background-color: #fafafa;">
                        """, unsafe_allow_html=True)
                    
                    # Store name and score
                    st.markdown(f"**🏪 {store_name}**")
                    st.markdown(f"**Score:** {review.get('criticality_score', 0)}/10")
                    
                    # Truncated review text
                    st.markdown(f"**Review:** {truncated_text}")
                    
                    # Action to be performed (highlighted)
                    st.markdown(f"**Action:** <span style='background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px;'>{action}</span>", unsafe_allow_html=True)
                    
                    # Action button
                    if st.button(f"⚡ Take Action", key=f"action_btn_{i}", type="primary" if is_selected else "secondary", use_container_width=True):
                        st.session_state.selected_review = i
                        st.rerun()
                    
                    st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.subheader("🚀 Action Execution")
            
            if st.session_state.selected_review is not None:
                selected_review = critical_reviews[st.session_state.selected_review]
                
                # Display selected review details
                st.markdown("**Selected Review Details:**")
                
                col2a, col2b = st.columns(2)
                with col2a:
                    st.write(f"**Customer:** {selected_review.get('customer_name', 'Anonymous')}")
                    st.write(f"**Store:** {selected_review.get('store_branch') or selected_review.get('store_name', 'Unknown')}")
                    st.write(f"**Criticality Score:** {selected_review.get('criticality_score', 0)}/10")
                    st.write(f"**Category:** {selected_review.get('complaint_category', 'N/A')}")
                    st.write(f"**Product:** {selected_review.get('complaint_product', 'N/A')}")
                
                with col2b:
                    st.write(f"**Emotional Intensity:** {selected_review.get('emotional_intensity', 'N/A')}/10")
                    st.write(f"**Lost Customer Risk:** {selected_review.get('lost_customer_risk', 'N/A')}/10")
                    st.write(f"**Social Media Risk:** {'Yes' if selected_review.get('social_amplification_risk') else 'No'}")
                    st.write(f"**Resolution Requested:** {selected_review.get('resolution_requested', 'N/A')}")
                
                # Full review text
                st.markdown("**Full Review Text:**")
                st.text_area("", value=selected_review.get('original_review', ''), height=100, disabled=True)
                
                # Action to be performed
                st.markdown("**Action to be Performed:**")
                st.info(selected_review.get('action_to_be_performed', 'No action specified'))
                
                # Execute action button
                if not st.session_state.executing_actions:
                    if st.button("⚡ Execute Action", type="primary", use_container_width=True):
                        if not st.session_state.openai_api_key:
                            st.error("Please configure your OpenAI API key first.")
                        else:
                            st.session_state.executing_actions = True
                            st.rerun()
                
                # Action execution area
                if st.session_state.executing_actions:
                    st.markdown("**Action Execution Log:**")
                    
                    # Create a placeholder for real-time updates
                    action_placeholder = st.empty()
                    
                    try:
                        # Convert to CriticalReview object
                        critical_review = CriticalReview(
                            review_id=selected_review.get('specific_issue', f"review_{st.session_state.selected_review}"),
                            store_name=selected_review.get('store_branch', 'Unknown'),
                            customer_name=selected_review.get('customer_name', 'Anonymous'),
                            review_text=selected_review.get('original_review', ''),
                            criticality_score=selected_review.get('criticality_score', 0),
                            action_to_be_performed=selected_review.get('action_to_be_performed', ''),
                            timestamp=datetime.datetime.now(),
                            customer_contact=selected_review.get('customer_contact'),
                            category=selected_review.get('complaint_category')
                        )
                        
                        # Initialize agent
                        agent3 = CriticalReviewAgent(st.session_state.openai_api_key)
                        
                        # Parse action plan
                        action_plan = agent3._parse_action_plan(critical_review.action_to_be_performed)
                        
                        # Execute actions one by one with real-time feedback
                        all_results = []
                        action_log = []
                        
                        for j, action_type in enumerate(action_plan):
                            # Show current action being executed
                            action_log.append(f"🔄 Executing: {action_type.value.replace('_', ' ').title()}")
                            
                            with action_placeholder.container():
                                for log_entry in action_log:
                                    st.write(log_entry)
                            
                            # Execute the action
                            result = agent3._execute_action(critical_review, action_type)
                            
                            if result:
                                all_results.append(result)
                                
                                # Show detailed result
                                if action_type == ActionType.RAISE_URGENT_TICKET:
                                    ticket_id = result.details.get('ticket_id', 'N/A')
                                    action_log.append(f"✅ Ticket Created: {ticket_id}")
                                
                                elif action_type == ActionType.ESCALATE_TO_MANAGER:
                                    email_draft = result.details.get('email_draft', {})
                                    action_log.append(f"✅ Email Drafted to: {email_draft.get('to', 'N/A')}")
                                    action_log.append(f"📧 Subject: {email_draft.get('subject', 'N/A')}")
                                
                                elif action_type == ActionType.SEND_APOLOGY_EMAIL:
                                    email_draft = result.details.get('email_draft', {})
                                    action_log.append(f"✅ Apology Email Drafted")
                                    action_log.append(f"📧 Subject: {email_draft.get('subject', 'N/A')}")
                                
                                elif action_type == ActionType.REQUEST_STAFF_TRAINING:
                                    training_focus = result.details.get('training_focus', 'N/A')
                                    action_log.append(f"✅ Training Request Sent")
                                    action_log.append(f"🎯 Focus: {training_focus}")
                                
                                else:
                                    action_log.append(f"✅ {action_type.value.replace('_', ' ').title()} Completed")
                                
                                # Update display
                                with action_placeholder.container():
                                    for log_entry in action_log:
                                        st.write(log_entry)
                                
                                # Small delay for better UX
                                time.sleep(1)
                        
                        # Mark as processed
                        selected_review['status'] = 'processed'
                        selected_review['action_taken'] = all_results
                        
                        # Store results
                        st.session_state.action_results = all_results
                        st.session_state.action_log = action_log
                        
                        # Final success message
                        action_log.append(f"🎉 All actions completed successfully!")
                        
                        with action_placeholder.container():
                            for log_entry in action_log:
                                st.write(log_entry)
                        
                        st.success(f"✅ Executed {len(all_results)} action(s) for review!")
                        
                        # Reset execution state
                        st.session_state.executing_actions = False
                        
                    except Exception as e:
                        st.error(f"Error executing actions: {str(e)}")
                        st.session_state.executing_actions = False
                
                # Show action results if available
                if st.session_state.action_results:
                    st.markdown("**Action Results:**")
                    for i, result in enumerate(st.session_state.action_results):
                        with st.expander(f"Action {i+1}: {result.action_type.value.replace('_', ' ').title()}"):
                            st.write(f"**Status:** {result.status}")
                            st.write(f"**Timestamp:** {result.timestamp}")
                            st.write(f"**Follow-up Required:** {result.follow_up_required}")
                            
                            if result.details:
                                st.write("**Details:**")
                                st.json(result.details)
            else:
                st.info("Select a review from the left panel to view details and take action.")
    
    st.divider()
    
    # Display action results
    st.subheader("📋 Action Results")
    
    if st.session_state.action_results:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        total_actions = len(st.session_state.action_results)
        follow_ups_needed = len([r for r in st.session_state.action_results if r.follow_up_required])
        
        with col1:
            st.metric("Total Actions", total_actions)
        with col2:
            st.metric("Follow-ups Needed", follow_ups_needed)
        with col3:
            completion_rate = ((total_actions - follow_ups_needed) / total_actions * 100) if total_actions > 0 else 0
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
        
        # Detailed action results
        st.subheader("📧 Email Communications")
        
        email_actions = [r for r in st.session_state.action_results 
                        if 'email_draft' in r.details]
        
        if email_actions:
            for i, action in enumerate(email_actions):
                with st.expander(f"Email {i+1}: {action.action_type.value}"):
                    email = action.details['email_draft']
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write("**To:**", email.get('to', 'N/A'))
                        st.write("**Subject:**", email.get('subject', 'N/A'))
                        st.write("**Status:**", action.status)
                        st.write("**Follow-up:**", "Yes" if action.follow_up_required else "No")
                    
                    with col2:
                        st.write("**Email Content:**")
                        st.text_area(
                            "Email Body",
                            value=email.get('body', 'No content available'),
                            height=200,
                            key=f"email_body_{i}",
                            disabled=True
                        )
        
        # Action type breakdown
        st.subheader("📊 Action Type Analysis")
        
        action_types = [r.action_type.value for r in st.session_state.action_results]
        action_counts = Counter(action_types)
        
        if action_counts:
            # Create pie chart
            fig = px.pie(
                values=list(action_counts.values()),
                names=list(action_counts.keys()),
                title="Distribution of Action Types"
            )
            st.plotly_chart(fig, use_container_width=True, key="action_types_chart")
        
        # Detailed action log
        st.subheader("📝 Detailed Action Log")
        
        action_data = []
        for action in st.session_state.action_results:
            action_data.append({
                'Action Type': action.action_type.value,
                'Status': action.status,
                'Follow-up Required': action.follow_up_required,
                'Timestamp': action.timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(action, 'timestamp') else 'N/A'
            })
        
        if action_data:
            df_actions = pd.DataFrame(action_data)
            st.dataframe(df_actions, use_container_width=True)
    
    else:
        st.info("No actions have been processed yet. Process critical reviews to see action results.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Customer Review Processor v1.0**")
st.sidebar.markdown("Built with Streamlit & OpenAI")