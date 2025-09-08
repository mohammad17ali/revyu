import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from collections import Counter
import json
import time

# Import from functions.py
try:
    from functions import (
        FashionFeedbackProcessor,
        process_store_list,
        calculate_satisfaction_scores,
        calculate_dissatisfaction_scores,
        get_common_issues,
        CriticalReview,
        ActionType,
        Priority,
        ActionResult
    )
    # Try importing CriticalReviewAgent if it exists in functions.py
    try:
        from functions import CriticalReviewAgent
    except ImportError:
        CriticalReviewAgent = None
        st.warning("CriticalReviewAgent not available. Action Center functionality will be limited.")
        
except ImportError as e:
    st.error(f"Error importing from functions.py: {str(e)}")
    st.error("Please ensure functions.py is in the same directory with all required classes and functions.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Customer Review Processor",
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
if 'google_maps_api_key' not in st.session_state:
    st.session_state.google_maps_api_key = ""

# Sidebar for navigation
st.sidebar.title("🛍️ Review Processor")
page = st.sidebar.selectbox("Select Page", ["Review Processor", "Dashboard", "Action Center"])

# API Key input in sidebar
with st.sidebar.expander("API Configuration", expanded=True):
    openai_api_key = st.text_input(
        "OpenAI API Key", 
        type="password", 
        value=st.session_state.openai_api_key,
        help="Required for AI-powered review analysis"
    )
    
    google_maps_api_key = st.text_input(
        "Google Maps API Key", 
        type="password", 
        value=st.session_state.google_maps_api_key,
        help="Required for fetching Google Reviews"
    )
    
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        st.success("✅ OpenAI API Key configured!")
        
    if google_maps_api_key:
        st.session_state.google_maps_api_key = google_maps_api_key
        st.success("✅ Google Maps API Key configured!")

# Helper function for processing single review
def process_single_review_wrapper(review_text, store_name, openai_api_key):
    """Wrapper function to process a single review"""
    try:
        processor = FashionFeedbackProcessor(openai_api_key)
        result = processor.process_single_review(review_text)
        result['store_name'] = store_name  # Add store name to result
        return result
    except Exception as e:
        st.error(f"Error processing review: {str(e)}")
        return None

# PAGE 1: REVIEW PROCESSOR
if page == "Review Processor":
    st.title("🔄 Review Processor Pipeline")
    st.markdown("Process customer reviews through our AI-powered analysis pipeline")
    
    # Check API keys
    if not st.session_state.openai_api_key:
        st.warning("⚠️ Please configure your OpenAI API key in the sidebar to proceed.")
        st.stop()
    
    if not st.session_state.google_maps_api_key:
        st.warning("⚠️ Please configure your Google Maps API key in the sidebar to proceed.")
        st.stop()
    
    tab1, tab2 = st.tabs(["🏪 Bulk Store Processing", "📝 Single Review Testing"])
    
    with tab1:
        st.header("Bulk Store Processing")
        st.markdown("Select multiple stores to process all their reviews through the pipeline")
        
        # Store selection
        available_stores = [
            'Zudio Koramangala', 'Zudio Phoenix Mumbai', 'Zudio Jammu', 
            'Zudio Bhopal', 'Zudio Velachery', 'Zudio Delhi', 
            'Zudio Pune', 'Zudio Hyderabad', 'Zudio Chennai'
        ]
        
        selected_stores = st.multiselect(
            "Select stores to process:",
            available_stores,
            default=['Zudio Bhopal', 'Zudio Velachery'],
            help="Select one or more stores to fetch and process their Google reviews"
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
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Initialize processor
                    status_text.text("Fetching and processing store reviews...")
                    progress_bar.progress(30)
                    
                    # Process stores using the imported function
                    positive_dataset, negative_dataset, critical_dataset = process_store_list(
                        selected_stores, 
                        processor, 
                        st.session_state.google_maps_api_key
                    )
                    
                    progress_bar.progress(90)
                    
                    # Store in session state
                    st.session_state.positive_dataset = positive_dataset
                    st.session_state.negative_dataset = negative_dataset
                    st.session_state.critical_dataset = critical_dataset
                    st.session_state.processed_data = True
                    
                    progress_bar.progress(100)
                    status_text.text("Processing complete!")
                    
                    # Display summary
                    st.success("✅ Processing completed successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Positive Reviews", len(positive_dataset))
                    with col2:
                        st.metric("Negative Reviews", len(negative_dataset))
                    with col3:
                        st.metric("Critical Reviews", len(critical_dataset))
                    
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    st.error("Please check your API keys and internet connection.")
    
    with tab2:
        st.header("Single Review Testing")
        st.markdown("Test the pipeline with a single review to see how it works")
        
        col1, col2 = st.columns(2)
        with col1:
            test_store = st.selectbox("Select store:", available_stores)
        with col2:
            sentiment_type = st.selectbox("Review Type:", ["Positive", "Negative", "Critical"])
        
        # Sample reviews based on type
        sample_reviews = {
            "Positive": "Amazing experience at this store! The staff was so helpful and friendly. Found exactly what I was looking for. The kurta collection is fantastic and very reasonably priced. Will definitely come back and recommend to friends!",
            "Negative": "Very disappointed with the service. The staff was rude and unhelpful. The fitting rooms were dirty and the clothes quality was poor. Waited for 30 minutes just to try on a dress. Not coming back here again.",
            "Critical": "Terrible experience! The store manager was extremely rude and refused to help with a return. The product was defective but they wouldn't acknowledge it. This is completely unacceptable. I'm posting this on social media to warn others."
        }
        
        review_text = st.text_area(
            "Enter review text:",
            value=sample_reviews[sentiment_type],
            height=100,
            help="Enter a customer review to process through the pipeline"
        )
        
        if st.button("🔍 Analyze Single Review"):
            if review_text:
                with st.spinner("Analyzing review..."):
                    try:
                        # Process single review
                        result = process_single_review_wrapper(
                            review_text, 
                            test_store, 
                            st.session_state.openai_api_key
                        )
                        
                        if result:
                            st.success("✅ Review processed successfully!")
                            
                            # Display results in an expandable format
                            with st.expander("📊 Analysis Results", expanded=True):
                                st.json(result)
                            
                            # Show classification
                            sentiment = result.get('sentiment', 'Unknown')
                            if sentiment == 'Positive':
                                st.success(f"Classification: {sentiment}")
                                score = result.get('satisfaction_score', 0)
                                st.metric("Satisfaction Score", f"{score}/10")
                            elif sentiment == 'Negative':
                                st.warning(f"Classification: {sentiment}")
                                score = result.get('criticality_score', 0)
                                st.metric("Criticality Score", f"{score}/10")
                                
                                if score > 7:
                                    st.error("⚠️ This review requires immediate attention!")
                    
                    except Exception as e:
                        st.error(f"Error processing review: {str(e)}")

# PAGE 2: DASHBOARD
elif page == "Dashboard":
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
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top 3
            st.markdown("**Top 3 Performers:**")
            for i, (store, score) in enumerate(sorted_stores[:3]):
                st.write(f"{i+1}. {store}: {score:.1f}/10")
        else:
            st.info("No positive reviews available for analysis")
    
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
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top 3 issues
            st.markdown("**Top 3 Areas for Improvement:**")
            for i, (store, score) in enumerate(sorted_stores[:3]):
                st.write(f"{i+1}. {store}: {score:.1f}/10")
        else:
            st.info("No negative reviews available for analysis")
    
    st.divider()
    
    # Common Issues Analysis
    st.subheader("🔍 Common Issues Analysis")
    st.markdown("Select stores to identify the most common customer complaints")
    
    # Store selector for issues analysis
    all_stores = list(set([r.get('store_name', 'Unknown') for r in (st.session_state.negative_dataset or [])]))
    if all_stores:
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
                st.plotly_chart(fig, use_container_width=True)
                
                # Display in table format
                st.markdown("**Issue Breakdown:**")
                df_issues = pd.DataFrame(common_issues, columns=['Issue', 'Frequency'])
                st.dataframe(df_issues, use_container_width=True)
            else:
                st.info("No common issues found for selected stores")
    else:
        st.info("No stores available for analysis")
    
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
            st.plotly_chart(fig, use_container_width=True)
    
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

# PAGE 3: ACTION CENTER
elif page == "Action Center":
    st.title("⚡ Action Center")
    st.markdown("Monitor and execute actions on critical reviews")
    
    if not st.session_state.processed_data:
        st.warning("Please process some reviews first using the Review Processor page.")
        st.stop()
    
    # Check if CriticalReviewAgent is available
    if not CriticalReviewAgent:
        st.error("Action Center functionality is not available. CriticalReviewAgent class is missing from functions.py")
        st.info("Please ensure the CriticalReviewAgent class and related classes are included in functions.py")
        st.stop()
    
    # Critical Review Processing Section
    st.subheader("🚨 Critical Review Processing")
    
    critical_reviews = [r for r in (st.session_state.critical_dataset or []) 
                       if r.get('criticality_score', 0) > 7]
    
    if not critical_reviews:
        st.info("No critical reviews requiring immediate action found.")
    else:
        st.warning(f"Found {len(critical_reviews)} critical reviews requiring action")
        
        # Display critical reviews
        with st.expander("View Critical Reviews", expanded=True):
            for i, review in enumerate(critical_reviews[:5]):  # Show first 5
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Review {i+1}:** {review.get('original_review', 'No text available')[:200]}...")
                    st.write(f"**Store:** {review.get('store_name', 'Unknown')}")
                    st.write(f"**Issue:** {review.get('specific_issue', 'Not specified')}")
                with col2:
                    st.metric("Criticality", f"{review.get('criticality_score', 0)}/10")
                st.divider()
        
        # Process critical reviews button
        if st.button("🤖 Process Critical Reviews", type="primary"):
            with st.spinner("Processing critical reviews..."):
                try:
                    agent3 = CriticalReviewAgent(st.session_state.openai_api_key)
                    all_results = []
                    
                    progress_bar = st.progress(0)
                    for i, review in enumerate(critical_reviews):
                        if review.get('status') != 'processed':
                            # Convert to CriticalReview object
                            critical_review = CriticalReview(
                                review_id=review.get('specific_issue', f"review_{i}"),
                                store_name=review.get('store_name', 'Unknown'),
                                customer_name=review.get('customer_name', 'Anonymous'),
                                review_text=review.get('original_review', ''),
                                criticality_score=review.get('criticality_score', 0),
                                action_to_be_performed=review.get('action_to_be_performed', ''),
                                timestamp=datetime.datetime.now()
                            )
                            
                            # Process the review
                            results = agent3.process_critical_review(critical_review)
                            
                            # Store results
                            review['status'] = 'processed'
                            review['action_taken'] = results
                            all_results.extend(results)
                        
                        progress_bar.progress((i + 1) / len(critical_reviews))
                    
                    # Store results in session state
                    st.session_state.action_results = all_results
                    
                    st.success(f"✅ Processed {len(critical_reviews)} critical reviews!")
                    
                except Exception as e:
                    st.error(f"Error processing critical reviews: {str(e)}")
    
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
            st.plotly_chart(fig, use_container_width=True)
        
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
        
        # Email Communications
        st.subheader("📧 Generated Communications")
        
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
    else:
        st.info("No actions have been processed yet. Process critical reviews to see action results.")

# Footer
# st.sidebar.markdown("---")
# st.sidebar.markdown("**Customer Review Processor v1.0**")
# st.sidebar.markdown("Built with Streamlit & OpenAI")
# st.sidebar.markdown("📁 Functions imported from `functions.py`")Initializing AI processor...")
#                     progress_bar.progress(10)
#                     processor = FashionFeedbackProcessor(st.session_state.openai_api_key)
                    
