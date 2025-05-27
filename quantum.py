import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests
import itertools
import time

# Set page config
st.set_page_config(page_title="Quantum Retail Toolkit", layout="wide")

# Load Lottie animation
@st.cache_data
def load_lottieurl(url):
    try: # Added try/except for robustness
        r = requests.get(url, timeout=5) # Added timeout
        if r.status_code != 200:
            st.error(f"Failed to load Lottie animation from {url}. Status code: {r.status_code}")
            return None
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading Lottie animation from {url}: {e}")
        return None

# Load Lottie animation - Keep only the ad Lottie as requested in the first code block structure
lottie_ad = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_9cyyl8i4.json")
# Load Lotties for other tabs for checks, as included in the fix for NameError


# Tabs layout
st.title("Quantum-Inspired Retail Toolkit")

st_lottie(lottie_ad, height=200, key="ad")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Campaign Optimizer",
    "🧠 Sentiment Scanner",
    "📋 Project Tracker",
    "🎯 Personalization Booster",
    "🚨 Anomaly Detector",
    "📍 Ad Placement Optimizer",
    "👥 Audience Prediction Engine",
    "🚚 Logistics & Scheduling",
    "🔮 Forecast Simulator",
    "⚡ Real-Time Bidding (RTB)"
])


# ========== Tab 1: Campaign Optimizer ==========
with tab1:
    st.header("Inventory & Campaign Optimization")

    # Simulated multi-channel campaign data
    np.random.seed(42)
    budgets = [5000, 10000, 15000]
    time_slots = ["Morning", "Afternoon", "Evening"]
    locations = ["Downtown", "Suburbs", "Mall", "Airport"]
    segments = ["Teens", "Adults", "Seniors"]

    # Original functions from the user's provided code
    def simulate_campaign_data():
        combos = list(itertools.product(budgets, time_slots, locations, segments))
        data = pd.DataFrame(combos, columns=["Budget", "Time Slot", "Location", "Segment"])
        data["Reach"] = np.random.randint(1000, 10000, len(data))
        data["Engagement"] = np.random.rand(len(data)) * 100
        return data

    def optimize_campaign(data, budget):
        filtered = data[data["Budget"] <= budget].copy()
        filtered["Score"] = filtered["Reach"] * filtered["Engagement"]
        top_combos = filtered.sort_values(by="Score", ascending=False).head(10)
        return top_combos

    st.sidebar.header("Campaign Filter")
    selected_budget = st.sidebar.selectbox("Select Maximum Budget:", budgets)

    st.subheader("All Campaign Possibilities")
    # Show only head(10) as in the original code
    data = simulate_campaign_data() # Re-call the function
    if not data.empty:
         st.dataframe(data.head(10))
    else:
         st.info("No campaign data generated.")


    optimized = optimize_campaign(data, selected_budget)

    st.subheader("Top Optimized Campaign Strategies")
    st.write(f"Showing top 10 options for budget <= {selected_budget}")
    if not optimized.empty:
        st.dataframe(optimized)

        st.subheader("Optimization Visualization")
        fig, ax = plt.subplots(figsize=(10, 4))
        optimized.plot(kind='bar', x='Location', y='Score', ax=ax, color='seagreen')
        plt.ylabel("Optimization Score")
        plt.title("Best Locations by Campaign Performance")
        st.pyplot(fig)
        plt.close(fig) # Added closing plot
    else:
        st.info(f"No optimized strategies found for budget <= {selected_budget}.")


    st.markdown("---")
    st.markdown("*Simulated Grover-like search for optimizing multi-channel retail campaigns.*")

# ========== Tab 2: Sentiment Scanner ==========
with tab2:
    st.header("Brand Sentiment Scanner")

    # Simulated customer feedback dataset
    np.random.seed(42)
    # Original data generation
    feedback = pd.DataFrame({
        "Review": [f"Feedback {i+1}" for i in range(1000)],
        "Sentiment Score": np.random.randn(1000)
    })

    # Original filter function
    def grover_simulation(data, threshold):
        hits = data[data['Sentiment Score'] < threshold]
        return hits

    threshold = st.slider("Sentiment Threshold", min_value=-3.0, max_value=0.0, step=0.1, value=-1.5)

    st.subheader("Sample Customer Feedback")
    # Show only sample(10) as in the original code
    if not feedback.empty:
        st.dataframe(feedback.sample(10))
    else:
         st.info("No feedback data generated.")


    result_df = grover_simulation(feedback, threshold)

    st.subheader(f"Detected Negative Feedback (Score < {threshold})")
    st.write(f"{len(result_df)} negative reviews detected.")
    if not result_df.empty:
        st.dataframe(result_df.head(10))
    else:
         st.info(f"No negative reviews detected below {threshold}.")


    st.subheader("Sentiment Distribution")
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    if not feedback.empty:
        feedback['Sentiment Score'].plot(kind='hist', bins=30, ax=ax2, color='salmon', alpha=0.7)
        plt.axvline(x=threshold, color='black', linestyle='--', label='Threshold')
        plt.legend()
        plt.title("Histogram of Sentiment Scores")
        st.pyplot(fig2)
        plt.close(fig2) # Added closing plot
    else:
        st.info("No data for sentiment distribution plot.")

    st.markdown("---")
    st.markdown("*Simulated Grover-like filter to surface brand-critical feedback efficiently.*")

# ========== Tab 3: Project Tracker ==========
with tab3:
    st.header("Campaign Project Tracker")

    # Raw data
    # Original data
    df = pd.DataFrame({
        "Year": [2023]*6 + [2024]*8,
        "Month Commissioned": ["July", "August", "September", "September", "December", "December", "January", "April", "July", "June", "October", "October", "November", "December"],
        "Client": ["Moving Walls"] * 14,
        "Brand": ["Samsung", "AirAsia", "Mudah", "KFC", "Coca-cola", "IHG Hotel", "Panasonic", "Cheetos", "Vivo", "Sprite", "Nestle", "Jollibee", "Cheetos", "M&M's"],
        "Project Name": ["BLS Vinyasa", "BLS Triplet", "BLSE 3", "BLSE 4", "Sparkling", "Comfort", "Air", "LATAM", "BLS Vivo", "BLS PH 1", "LATAM 2", "BLS PH 2", "LATAM 3", "BLS PH 3"],
        "Project Category": ["Smartphones", "Airlines", "Marketplace", "Restaurants", "RTD Beverage", "Hotel", "Consumer Goods", "Snacks", "Smartphones", "RTD Beverage", "Baby product", "Restaurants", "Snacks", "Snacks"],
        "Market": ["Indonesia", "Malaysia", "Malaysia", "Malaysia", "Philippines", "Thailand, Australia", "Malaysia", "Brazil", "Indonesia", "Philippines", "Brazil", "Philippines", "Brazil", "Philippines"],
        "Total Sample": [300, 300, 300, 300, 300, 300, 300, 300, None, 300, 300, 300, 300, 300]
    })

    # Filters - Original placement in sidebar
    with st.sidebar:
        st.subheader("Project Filters")
        year_filter = st.multiselect("Select Year(s):", sorted(df["Year"].unique()), default=sorted(df["Year"].unique()))
        brand_filter = st.multiselect("Select Brand(s):", sorted(df["Brand"].unique()), default=sorted(df["Brand"].unique()))

    filtered_df = df[df["Year"].isin(year_filter) & df["Brand"].isin(brand_filter)]

    st.subheader("Filtered Project Table")
    st.dataframe(filtered_df) # Display filtered dataframe

    st.subheader("Total Sample by Market")
    # Original grouping and display
    summary = filtered_df.groupby("Market")["Total Sample"].sum().reset_index().dropna()
    st.dataframe(summary)

    st.subheader("Sample Distribution")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    if not summary.empty: # Added check
        summary.set_index("Market")["Total Sample"].plot(kind="bar", ax=ax3, color="steelblue")
        plt.ylabel("Total Sample Size")
        plt.title("Sample Size by Market")
        st.pyplot(fig3)
        plt.close(fig3) # Added closing plot
    else:
        st.info("No data for Sample Distribution plot.")


    st.markdown("---")
    st.markdown("*Project tracking powered by filters and charts for fast data insights.*")

# ========== Tab 4: Targeting Booster ==========
with tab4:
    st.header("Personalization & Targeting Booster")

    st.markdown("""
    **Amplitude Amplification for Relevance Boosting**

    This module simulates the use of amplitude amplification in a quantum recommender system — where the probability of relevant promotions being selected is boosted.
    """)

    np.random.seed(42)
    promo_items = ["10% off", "Free Sample", "Buy 1 Get 1", "VIP Invite", "Limited Edition"]
    probabilities = np.random.rand(len(promo_items))
    probabilities /= probabilities.sum()

    def amplify_probabilities(probs):
        amplified = probs**2
        return amplified / amplified.sum()

    st.subheader("Original Promotion Probabilities")
    df_probs = pd.DataFrame({"Promotion": promo_items, "Probability": probabilities})
    st.dataframe(df_probs)

    amplified_probs = amplify_probabilities(probabilities)
    df_amp = pd.DataFrame({"Promotion": promo_items, "Amplified Probability": amplified_probs})

    st.subheader("Boosted Promotion Probabilities")
    st.dataframe(df_amp)

    st.subheader("Relevance Boost Visualization") # Updated plot title in header
    fig4, ax4 = plt.subplots(figsize=(8, 4)) # Increased size
    df_amp.set_index("Promotion")["Amplified Probability"].plot(kind='bar', ax=ax4, color="purple")
    plt.ylabel("Boosted Probability")
    plt.title("Relevance Boost via Amplitude Amplification")
    st.pyplot(fig4)
    plt.close(fig4) # Added closing plot


# ========== Tab 5: Anomaly Detector ==========
with tab5:
    st.header("Fraud & Anomaly Detector")

    np.random.seed(0)
    transaction_ids = [f"TX-{i:04d}" for i in range(1000)]
    values = np.random.normal(loc=100, scale=20, size=1000)
    values[::50] += np.random.normal(loc=300, scale=50, size=20)  # Inject anomalies every 50th row

    df_anomaly = pd.DataFrame({"Transaction ID": transaction_ids, "Value": values})

    mean = df_anomaly["Value"].mean()
    std = df_anomaly["Value"].std()
    threshold = mean + 3 * std

    anomalies = df_anomaly[df_anomaly["Value"] > threshold]

    st.subheader("Anomalous Transactions Detected")
    if not anomalies.empty: # Added check
        st.dataframe(anomalies)
    else:
         st.info("No anomalies detected above the default threshold.")


    st.subheader("Transaction Value Distribution")
    fig5, ax5 = plt.subplots(figsize=(8, 4)) # Increased size
    if not df_anomaly.empty: # Added check
        df_anomaly["Value"].plot(kind='hist', bins=30, ax=ax5, color='orange', alpha=0.6)
        plt.axvline(x=threshold, color='red', linestyle='--', label=f'Anomaly Threshold ({threshold:.2f})') # Added threshold value to label
        plt.title("Histogram of Transaction Values")
        plt.legend()
        st.pyplot(fig5)
        plt.close(fig5) # Added closing plot
    else:
        st.info("No data for Transaction Value Distribution plot.")


# ========== Tab 6: Ad Placement Optimizer ==========
with tab6:
    st.header("📍 Ad Placement Optimizer")
    # Using lottie_opt from general loads, add check
   
    st.markdown("""
    **Quantum-Inspired Optimization for Ad Placements**

    Deciding where and when to place ads across a network of screens (digital billboards, in-store displays, online banners) is a complex optimization problem. Given constraints like budget, target audience, and location availability, the goal is to find the *optimal combination* of placements to maximize reach, engagement, or conversions. This is a classic combinatorial optimization problem, which quantum annealing or QAOA (Quantum Approximate Optimization Algorithm) are being explored to solve.

    This simulation generates a list of possible placements with simulated costs and effectiveness, and allows you to find combinations that fit within a budget and maximize a simple score.
    """)

    np.random.seed(46)
    # Define lists OUTSIDE the function, but inside the tab's scope
    placement_types = ["Digital Billboard", "Mall Kiosk Screen", "Bus Shelter Display", "Retail Store TV", "Online Ad Spot"]
    locations_p = ["CBD", "Suburbs (North)", "Suburbs (South)", "Shopping Mall A", "Shopping Mall B", "Downtown Store", "Neighborhood Store"]
    time_slots_p = ["Morning Peak", "Lunchtime", "Afternoon", "Evening Peak", "Late Night"]

    @st.cache_data # Add caching
    def generate_placement_data(p_types, locs, t_slots): # Pass lists as arguments
        placement_options = []
        for i, (ptype, loc, tslot) in enumerate(itertools.product(p_types, locs, t_slots)):
            cost = np.random.uniform(50, 500) + (p_types.index(ptype) * 50)
            reach = np.random.randint(100, 5000) + (locs.index(loc) * 200) + (t_slots.index(tslot) * 100)
            engagement_rate = np.random.uniform(0.01, 0.1) + (p_types.index(ptype) * 0.005)
            optimization_score = reach * engagement_rate

            placement_options.append({
                "ID": f"P{i+1:03d}",
                "Type": ptype,
                "Location": loc,
                "Time Slot": tslot,
                "Cost": round(cost, 2),
                "Simulated Reach": reach,
                "Simulated Engagement Rate": round(engagement_rate, 4),
                "Optimization Score": round(optimization_score, 2)
            })
        return pd.DataFrame(placement_options)

    df_placements = generate_placement_data(placement_types, locations_p, time_slots_p) # Call with the lists

    st.subheader("Available Ad Placement Options")
    st.dataframe(df_placements.sample(min(15, len(df_placements))).reset_index(drop=True))

    st.subheader("Find Optimal Placement Combinations")
    st.write("Select your maximum budget and how many top options you want to see. The tool simulates finding combinations that fit within a budget and maximize a simple score.")

    placement_budget = st.slider("Maximum Total Budget for Placements:", min_value=500.0, max_value=10000.0, step=100.0, value=5000.0)
    num_top_placements = st.slider("Show Top N Individual Placements:", 5, 30, 10)

    feasible_placements = df_placements[df_placements["Cost"] <= placement_budget].copy()

    if feasible_placements.empty:
         st.warning(f"No individual placements found within the budget of ${placement_budget:.2f}. Try increasing the budget.")
    else:
        top_individual_placements = feasible_placements.sort_values(by="Optimization Score", ascending=False).head(num_top_placements)

        st.subheader(f"Top {len(top_individual_placements)} Individual Placements (Cost <= ${placement_budget:.2f})")
        st.dataframe(top_individual_placements)

        st.subheader("Optimization Score Distribution of Top Placements")
        fig6, ax6 = plt.subplots(figsize=(12, 5))
        top_individual_placements.set_index("ID")["Optimization Score"].plot(kind="bar", ax=ax6, color='purple')
        ax6.set_ylabel("Optimization Score (Reach * Engagement Rate)")
        ax6.set_title("Optimization Score for Top Individual Ad Placements")
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6) # Add plot closing

    st.markdown("---")
    st.markdown("*Illustrates ad placement optimization, where quantum algorithms could accelerate finding the best combinations of placements under budget constraints.*")

# ========== Tab 7: Audience Prediction Engine ==========
with tab7:
    st.header("👥 Audience Prediction Engine")
    # Using lottie_ai from general loads, add check
    
    st.markdown("""
    **Quantum Machine Learning for Audience Forecasting**

    Predicting where and when specific audience segments will be present requires analyzing complex, multidimensional datasets (location data, time of day, events, demographics, etc.). Quantum machine learning algorithms, such as quantum neural networks or quantum clustering, could potentially identify subtle patterns in this data more effectively, leading to more accurate audience predictions for DOOH or targeted mobile ads.

    This simulation shows predicted audience presence at different locations and times based on hypothetical data.
    """)

    np.random.seed(47)
    # Define lists OUTSIDE the function, but inside the tab's scope
    audience_segments_p = ["Young Adults (18-30)", "Families with Kids", "Business Travelers", "Seniors (60+)"]
    locations_p = ["CBD", "Suburbs (North)", "Suburbs (South)", "Shopping Mall A", "Airport", "Community Park"]
    times_p = ["Morning (6-9)", "Late Morning (9-12)", "Lunch (12-2)", "Afternoon (2-5)", "Evening (5-8)", "Late Evening (8-11)"]
    days_p = ["Weekday", "Weekend"]

    @st.cache_data # Add caching
    def generate_prediction_data(segments, locs, times, days): # Pass lists as arguments
        prediction_data = []
        for seg, loc, tslot, day in itertools.product(segments, locs, times, days):
            predicted_presence = np.random.uniform(0, 100)

            if day == "Weekday":
                if loc == "CBD" and tslot in ["Morning (6-9)", "Late Morning (9-12)", "Afternoon (2-5)", "Evening (5-8)"]: predicted_presence += 50
                if loc == "Airport" and seg == "Business Travelers": predicted_presence += 80
                if loc in ["Suburbs (North)", "Suburbs (South)"] and tslot in ["Evening (5-8)", "Late Evening (8-11)"]: predicted_presence += 40
            elif day == "Weekend":
                if loc in ["Shopping Mall A", "Community Park"] and tslot in ["Late Morning (9-12)", "Lunch (12-2)", "Afternoon (2-5)"]: predicted_presence += 70
                if seg == "Families with Kids" and loc == "Community Park": predicted_presence += 60
                if loc == "CBD" and tslot == "Late Evening (8-11)": predicted_presence += 30

            predicted_presence = max(0, min(100, predicted_presence + np.random.normal(0, 15)))

            prediction_data.append({
                "Segment": seg,
                "Location": loc,
                "Time Slot": tslot,
                "Day Type": day,
                "Predicted Audience Presence (%)": round(predicted_presence, 1)
            })
        return pd.DataFrame(prediction_data)

    df_prediction = generate_prediction_data(audience_segments_p, locations_p, times_p, days_p) # Call with lists

    st.subheader("Explore Audience Presence Predictions")
    st.write("Filter the data to see predicted audience presence for specific segments, locations, times, and days.")

    col1_pred, col2_pred, col3_pred = st.columns(3)
    with col1_pred:
        selected_segment = st.multiselect("Select Segment(s):", sorted(df_prediction["Segment"].unique()), default=sorted(df_prediction["Segment"].unique()))
    with col2_pred:
        selected_location = st.multiselect("Select Location(s):", sorted(df_prediction["Location"].unique()), default=sorted(df_prediction["Location"].unique()))
    with col3_pred:
        selected_day = st.multiselect("Select Day Type(s):", sorted(df_prediction["Day Type"].unique()), default=sorted(df_prediction["Day Type"].unique()))

    filtered_prediction_df = df_prediction[
        df_prediction["Segment"].isin(selected_segment) &
        df_prediction["Location"].isin(selected_location) &
        df_prediction["Day Type"].isin(selected_day)
    ].reset_index(drop=True)

    st.dataframe(filtered_prediction_df)

    if not filtered_prediction_df.empty:
        st.subheader("Predicted Presence by Location and Time Slot (Filtered Data)")

        # Now times_p is defined outside the function and accessible here
        pivot_pred = filtered_prediction_df.pivot_table(
            index='Location',
            columns='Time Slot',
            values='Predicted Audience Presence (%)',
            aggfunc='mean'
        ).reindex(columns=times_p) # Ensure time slots order correctly

        if not pivot_pred.empty:
            fig7, ax7 = plt.subplots(figsize=(12, len(pivot_pred)*0.6 + 2))
            cax = ax7.matshow(pivot_pred, cmap='viridis')
            fig7.colorbar(cax, label='Avg. Predicted Presence (%)')

            ax7.set_xticks(np.arange(len(pivot_pred.columns)))
            ax7.set_yticks(np.arange(len(pivot_pred.index)))
            ax7.set_xticklabels(pivot_pred.columns, rotation=90, ha='left')
            ax7.set_yticklabels(pivot_pred.index)

            ax7.set_xlabel("Time Slot")
            ax7.set_ylabel("Location")
            ax7.set_title("Average Predicted Audience Presence (%)")
            plt.tight_layout()
            st.pyplot(fig7)
            plt.close(fig7) # Add plot closing
        else:
            st.info("No data to plot for the selected filters.")
    else:
        st.info("No data available for the selected filters.")

    st.markdown("---")
    st.markdown("*Illustrates audience prediction, where quantum machine learning could find deeper patterns in complex data for better forecasting.*")

# ========== Tab 8: Logistics & Scheduling ==========
with tab8:
    st.header("🚚 Logistics & Scheduling")
    # Using lottie_truck from general loads, add check
    

    st.markdown("""
    **Quantum Routing for Physical Ad Logistics**

    Managing the logistics of physical ad placements (like installing new billboards, performing maintenance, or delivering materials) involves complex routing and scheduling problems, often known as variants of the Traveling Salesperson Problem (TSP) or Vehicle Routing Problem (VRP). Quantum algorithms, particularly quantum annealing, are being researched for their potential to find optimal or near-optimal solutions to these NP-hard problems faster than classical methods for certain problem sizes.

    This simulation lists hypothetical logistics tasks and allows sorting, representing the input to a potential quantum routing optimizer.
    """)

    np.random.seed(48)
    # Define lists OUTSIDE the function, but inside the tab's scope
    task_types = ["Installation", "Maintenance", "Repair", "Material Delivery"]
    locations_l = ["Warehouse A", "Site 1 (Downtown)", "Site 2 (Suburbs)", "Site 3 (Mall)", "Site 4 (Airport)", "Site 5 (Industrial Park)"]
    priorities_l = ["High", "Medium", "Low"]
    statuses_l = ["Pending", "In Progress", "Completed", "Scheduled"]

    @st.cache_data # Add caching
    def generate_logistics_data(t_types, locs, priorities, statuses): # Pass lists as arguments
        logistics_tasks = []
        for i in range(30): # Simulate 30 tasks
            task_type = np.random.choice(t_types)
            start_loc_options = [loc for loc in locs if loc != "Warehouse A"] if task_type != "Material Delivery" else ["Warehouse A"]
            start_loc = np.random.choice(start_loc_options)
            end_loc_options = [loc for loc in locs if loc != start_loc] if task_type != "Material Delivery" else locs
            end_loc = np.random.choice(end_loc_options)

            priority = np.random.choice(priorities, p=[0.3, 0.5, 0.2])
            estimated_duration_hours = np.random.uniform(1, 8) + (priorities.index(priority) * 0.5)
            status = np.random.choice(statuses, p=[0.6, 0.2, 0.15, 0.05])

            logistics_tasks.append({
                "Task ID": f"LOG-{i+1:02d}",
                "Task Type": task_type,
                "Start Location": start_loc,
                "End Location": end_loc,
                "Priority": priority,
                "Estimated Duration (hours)": round(estimated_duration_hours, 1),
                "Status": status
            })
        return pd.DataFrame(logistics_tasks)

    df_logistics = generate_logistics_data(task_types, locations_l, priorities_l, statuses_l) # Call with lists

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    df_logistics['Priority_Order'] = df_logistics['Priority'].map(priority_order)

    st.subheader("Logistics Task List")
    st.write("A list of hypothetical tasks for physical ad logistics (installations, maintenance, etc.).")

    col1_log, col2_log, col3_log = st.columns(3)
    with col1_log:
        selected_task_type = st.multiselect("Filter by Task Type:", sorted(df_logistics["Task Type"].unique()), default=sorted(df_logistics["Task Type"].unique()))
    with col2_log:
        selected_status = st.multiselect("Filter by Status:", sorted(df_logistics["Status"].unique()), default=["Pending", "Scheduled", "In Progress"])
    with col3_log:
        sort_option = st.selectbox("Sort By:", ["Priority", "Estimated Duration (hours)", "Task ID"])

    filtered_logistics_df = df_logistics[
        df_logistics["Task Type"].isin(selected_task_type) &
        df_logistics["Status"].isin(selected_status)
    ].copy()

    if sort_option == "Priority":
        sorted_logistics_df = filtered_logistics_df.sort_values(by=["Priority_Order", "Estimated Duration (hours)"], ascending=[True, False])
    else:
        sorted_logistics_df = filtered_logistics_df.sort_values(by=sort_option, ascending=True)

    sorted_logistics_df_display = sorted_logistics_df.drop(columns=['Priority_Order'])

    st.dataframe(sorted_logistics_df_display.reset_index(drop=True))

    if not filtered_logistics_df.empty:
        st.subheader("Task Distribution by Type and Status (Filtered Data)")
        fig8, ax8 = plt.subplots(figsize=(10, 5))
        logistics_pivot = filtered_logistics_df.groupby(["Task Type", "Status"]).size().unstack(fill_value=0)
        logistics_pivot.plot(kind='bar', stacked=True, ax=ax8, cmap='viridis')
        ax8.set_ylabel("Number of Tasks")
        ax8.set_title("Logistics Tasks by Type and Status (Filtered Data)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig8)
        plt.close(fig8) # Add plot closing

    st.markdown("---")
    st.markdown("*Illustrates logistics and scheduling, where quantum annealing could potentially find optimal routes and schedules for complex physical tasks.*")

# ========== Tab 9: Forecast Simulator ==========
with tab9:
    st.header("🔮 Forecast Simulator")
    
    st.markdown("""
    **Quantum-Enhanced Forecasting & Scenario Planning**

    Forecasting the outcome of ad campaigns involves dealing with uncertainty and complex interactions between various factors (channels, budget, target audience, external events, competition). Quantum probabilistic simulations or quantum enhanced Monte Carlo methods could potentially model these uncertainties and complex dependencies more accurately and efficiently, allowing for better "what-if" scenario analysis and ROI forecasting.

    This simulation lets you adjust parameters and see a hypothetical forecast based on simple rules and randomness, representing the output of such a simulator.
    """)

    np.random.seed(49)

    st.subheader("Simulate Campaign Outcome")
    st.write("Adjust the parameters below to see how a hypothetical campaign outcome might be forecasted.")

    # Simulation inputs
    col1_fc, col2_fc = st.columns(2)
    with col1_fc:
        sim_budget = st.slider("Simulated Budget ($):", min_value=10000, max_value=100000, step=5000, value=50000)
        sim_channel = st.selectbox("Primary Channel:", ["DOOH", "Social Media", "Email", "In-Store Displays", "Programmatic Display"])
        sim_duration = st.slider("Campaign Duration (Weeks):", min_value=1, max_value=12, value=4)
    with col2_fc:
        sim_segment = st.selectbox("Target Segment:", ["Young Adults", "Families", "Business Professionals", "General Public"])
        sim_competitor_activity = st.slider("Simulated Competitor Activity (Low=1, High=5):", min_value=1, max_value=5, value=3)
        sim_market_sentiment = st.slider("Simulated Market Sentiment (Negative=1, Positive=5):", min_value=1, max_value=5, value=3)

    # Define constants OUTSIDE the function, inside the tab's scope
    base_impressions_per_dollar = {
        "DOOH": 15, "Social Media": 25, "Email": 5, "In-Store Displays": 10, "Programmatic Display": 20
    }
    base_conversion_rate = {
        "Young Adults": 0.015, "Families": 0.02, "Business Professionals": 0.01, "General Public": 0.018
    }
    channel_efficiency = {
         "DOOH": 1.2, "Social Media": 1.5, "Email": 1.8, "In-Store Displays": 1.3, "Programmatic Display": 1.1
    }

    @st.cache_data(show_spinner=False) # Add caching
    def simulate_outcome(budget, channel, duration, segment, competitor_activity, market_sentiment, # Pass inputs
                         imp_per_dollar, conv_rate, chan_eff): # Pass constants
        base_impressions = budget * imp_per_dollar.get(channel, 10)
        base_conversions = base_impressions * conv_rate.get(segment, 0.015)

        duration_factor = duration / 4
        market_sentiment_factor = (1 + (market_sentiment - 3) * 0.05)
        comp_impact = (1 - (competitor_activity - 1) * 0.05)

        adjusted_impressions = base_impressions * duration_factor * chan_eff.get(channel, 1.0) * max(0.1, market_sentiment_factor) * max(0.1, comp_impact)
        adjusted_conversions = base_conversions * duration_factor * chan_eff.get(channel, 1.0) * max(0.1, market_sentiment_factor * 1.2) * max(0.1, comp_impact * 0.8)

        final_impressions = max(0, adjusted_impressions + np.random.normal(0, adjusted_impressions * 0.1))
        final_conversions = max(0, adjusted_conversions + np.random.normal(0, adjusted_conversions * 0.2))

        simulated_revenue = final_conversions * 50
        simulated_roi = (simulated_revenue - budget) / budget if budget > 0 else 0

        return {
            "Simulated Impressions": int(final_impressions),
            "Simulated Conversions": int(final_conversions),
            "Simulated Revenue": round(simulated_revenue, 2),
            "Simulated ROI": round(simulated_roi, 3)
        }

    simulated_result = simulate_outcome(
        sim_budget, sim_channel, sim_duration, sim_segment, sim_competitor_activity, sim_market_sentiment,
        base_impressions_per_dollar, base_conversion_rate, channel_efficiency # Pass constants
    )

    st.subheader("Simulated Campaign Outcome Forecast")
    st.write("Based on the selected parameters:")

    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
    col_res1.metric("Impressions", f"{simulated_result['Simulated Impressions']:,}")
    col_res2.metric("Conversions", f"{simulated_result['Simulated Conversions']:,}")
    col_res3.metric("Revenue", f"${simulated_result['Simulated Revenue']:,}")
    col_res4.metric("ROI", f"{simulated_result['Simulated ROI'] * 100:.1f}%")

    st.subheader("Impact of Factors on Outcome (Illustrative)")
    st.write("This is a simplified model. A real quantum simulator would model complex interactions.")

    impact_data = {
        "Factor": ["Budget", "Duration", "Channel Efficiency", "Segment Relevance", "Competitor Activity", "Market Sentiment"],
        "Simulated Influence (relative)": [
            np.log(sim_budget) if sim_budget > 0 else 0,
            sim_duration,
            channel_efficiency.get(sim_channel, 1.0),
            base_conversion_rate.get(sim_segment, 0.015) * 100,
            (6 - sim_competitor_activity),
            sim_market_sentiment
        ]
    }
    df_impact = pd.DataFrame(impact_data)

    fig9, ax9 = plt.subplots(figsize=(10, 5))
    df_impact.set_index("Factor")["Simulated Influence (relative)"].plot(kind='bar', ax=ax9, color='teal')
    ax9.set_ylabel("Simulated Influence Score (Relative)")
    ax9.set_title("Hypothetical Influence of Factors on Campaign Outcome")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig9)
    plt.close(fig9) # Add plot closing

    st.markdown("---")
    st.markdown("*Illustrates forecasting, where quantum simulations could model campaign outcomes under uncertainty and complex dependencies more effectively.*")

# ========== Tab 10: Real-Time Bidding (RTB) ==========
with tab10:
    st.header("⚡ Real-Time Bidding (RTB) Engine")
    # Using lottie_speed from general loads, add check
    
    st.markdown("""
    **Quantum Speedups for Programmatic DOOH RTB**

    In Real-Time Bidding (RTB) for digital ad space, including Digital Out-of-Home (DOOH), decisions must be made in milliseconds. For complex bidding strategies involving numerous parameters (user data, context, inventory type, historical performance, competitor bids), classical algorithms face latency constraints. Quantum algorithms could potentially evaluate complex bid scenarios and constraints much faster, enabling more sophisticated and profitable bidding strategies within the strict time limits.

    This simulation shows incoming bid requests and a simplified "quantum-speedup" decision process to select which ones to bid on based on a set of criteria.
    """)

    np.random.seed(50)

    # Define constants OUTSIDE the function, inside the tab's scope
    inventory_types_rtb = ["Digital Billboard", "Transit Screen", "Retail TV", "Airport Display", "Street Poster (Digital)"]
    locations_rtb = ["CBD", "Mall A", "Airport Zone B", "City Square", "Business Park", "University Campus"]
    user_segments_rtb = ["Shopper", "Commuter", "Traveler", "Local Resident", "Student", "Office Worker"]
    min_bid_prices_rtb = {"Digital Billboard": 5.0, "Transit Screen": 3.0, "Retail TV": 2.5, "Airport Display": 7.0, "Street Poster (Digital)": 4.0}
    strategic_values_rtb = {"High": 3, "Medium": 2, "Low": 1}


    @st.cache_data # Add caching
    # Corrected parameter order: non-default parameters first
    def generate_bid_request(inv_types, locs, segments, min_prices, strategic_vals,
                             request_id_start=0, num_requests=10): # Parameters with defaults last

        bid_requests = []
        for i in range(request_id_start, request_id_start + num_requests):
            inv_type = np.random.choice(inv_types)
            loc = np.random.choice(locs)
            segment = np.random.choice(segments)
            min_price = min_prices.get(inv_type, 3.0) + np.random.uniform(-1.5, 1.5)
            context_score = np.random.uniform(1, 5)

            strategic_importance = np.random.choice(list(strategic_vals.keys()), p=[0.25, 0.5, 0.25])
            strategic_factor = strategic_vals[strategic_importance]

            calculated_value = (np.random.uniform(5, 20) + context_score * 2 + strategic_factor * 5 +
                               inv_types.index(inv_type) * 1 + segments.index(segment) * 0.5) * np.random.uniform(0.9, 1.1)

            bid_requests.append({
                "Request ID": f"REQ-{i:04d}",
                "Inventory Type": inv_type,
                "Location": loc,
                "Target Segment": segment,
                "Minimum Price (CPM)": round(max(1.0, min_price), 2),
                "Simulated Value Score": round(max(1.0, calculated_value), 2),
                "Strategic Importance": strategic_importance
            })
        return pd.DataFrame(bid_requests)

    num_requests_to_show = st.slider("Number of Incoming Bid Requests to Simulate:", min_value=5, max_value=50, value=15)

    # State management for data and decisions
    if 'last_request_id_rtb' not in st.session_state: # Unique key
        st.session_state.last_request_id_rtb = 0
    if 'bid_requests_df_rtb' not in st.session_state: # Unique key
        st.session_state.bid_requests_df_rtb = pd.DataFrame()
    if 'decisions_made_rtb' not in st.session_state: # Unique key
        st.session_state.decisions_made_rtb = False

    if st.button("Generate New Bid Requests", key="generate_bids_btn_rtb"): # Added unique key
         new_requests_df = generate_bid_request(inventory_types_rtb, locations_rtb, user_segments_rtb, min_bid_prices_rtb, strategic_values_rtb, # Pass constants first
                                                st.session_state.last_request_id_rtb, num_requests_to_show) # Pass inputs last
         st.session_state.bid_requests_df_rtb = new_requests_df # Use unique key
         st.session_state.last_request_id_rtb += num_requests_to_show # Use unique key
         st.session_state.decisions_made_rtb = False # Reset decisions, use unique key

    if not st.session_state.bid_requests_df_rtb.empty: # Use unique key
        st.subheader("Simulated Incoming Bid Requests")
        st.write("This table shows hypothetical bid requests received by an RTB system.")
        st.dataframe(st.session_state.bid_requests_df_rtb) # Use unique key
    else:
        st.info("Click 'Generate New Bid Requests' to start the simulation.")

    st.subheader("Quantum-Inspired Bid Decision")
    st.write("""
    In milliseconds, an RTB engine must decide:
    1. Is this impression relevant/valuable enough? (Based on value score, target segment match, etc.)
    2. Can we afford the minimum price?
    3. Does it meet strategic goals (e.g., high strategic importance)?
    4. What is the optimal bid price (not shown here)?

    Quantum speedups could evaluate complex criteria like these much faster. This simulation applies simple decision rules quickly.
    """)

    # Simple simulated decision logic
    @st.cache_data(show_spinner="Processing decisions (simulating quantum speedup)...") # Add caching and spinner
    def make_bid_decision(request_df, budget_per_request, min_value_score, require_strategic, strategic_vals): # Pass constants
        # Simulate processing delay
        # time.sleep(0.05) # A tiny delay

        decisions = []
        decision_reasons = []
        # Default to lowest strategic value if None is selected, so 'Any' means >= Low
        required_strategic_value = strategic_vals.get(require_strategic, min(strategic_vals.values())) if require_strategic is not None else min(strategic_vals.values())


        for index, row in request_df.iterrows():
            can_afford = row["Minimum Price (CPM)"] <= budget_per_request
            is_valuable = row["Simulated Value Score"] >= min_value_score
            # Check if strategic level meets or exceeds required level
            is_strategic = strategic_vals.get(row["Strategic Importance"], 0) >= required_strategic_value

            decision = "Bid" if can_afford and is_valuable and is_strategic else "Skip"

            reason = []
            if not can_afford: reason.append("Too Expensive")
            if not is_valuable: reason.append("Low Value")
            # Only add strategic reason if a strategic level was *required* (i.e., not None)
            if require_strategic is not None and not is_strategic: reason.append(f"Not Strategic Enough ({row['Strategic Importance']})")
            reason_str = ", ".join(reason) if reason else "Meets Criteria"

            decisions.append(decision)
            decision_reasons.append(reason_str)

        request_df_decisions = request_df.copy()
        request_df_decisions['Decision'] = decisions
        request_df_decisions['Decision Reason'] = decision_reasons

        return request_df_decisions

    st.subheader("Adjust Decision Parameters")
    st.write("These parameters control the simulated decision rules. A quantum system could handle more complex, dynamic parameters.")
    col_params1, col_params2, col_params3 = st.columns(3)
    with col_params1:
        budget_per_req = st.number_input("Max Budget Per Request (CPM):", min_value=1.0, max_value=20.0, value=6.0, step=0.5, key="budget_per_req_rtb") # Unique key
    with col_params2:
        min_val_score = st.number_input("Min Required Value Score:", min_value=5.0, max_value=30.0, value=15.0, step=1.0, key="min_val_score_rtb") # Unique key
    with col_params3:
        # Options for strategic importance include None ("Any")
        strategic_options = sorted(list(strategic_values_rtb.keys()) + [None], key=lambda x: (0 if x is None else strategic_values_rtb[x], x if x is not None else ''))
        req_strategic = st.selectbox("Minimum Strategic Importance:", strategic_options, index=0, format_func=lambda x: "Any" if x is None else x, key="req_strategic_rtb") # Unique key


    if not st.session_state.bid_requests_df_rtb.empty: # Use unique key
        # Use a unique key for the Run Decisions button
        if st.button("Run Bid Decisions", key="run_decisions_btn_rtb"): # Unique key
             st.session_state.bid_requests_df_decided_rtb = make_bid_decision( # Use unique key
                 st.session_state.bid_requests_df_rtb, # Use unique key
                 st.session_state.budget_per_req_rtb, # Use unique keys
                 st.session_state.min_val_score_rtb, # Use unique keys
                 st.session_state.req_strategic_rtb, # Use unique keys
                 strategic_values_rtb # Pass constants
             )
             st.session_state.decisions_made_rtb = True # Mark decisions as run, use unique key

        # Display results only if decisions have been made for the current data
        if st.session_state.get("decisions_made_rtb", False) and 'bid_requests_df_decided_rtb' in st.session_state: # Use unique keys
            st.subheader("Bid Decision Outcomes")
            st.dataframe(st.session_state.bid_requests_df_decided_rtb) # Use unique key

            st.subheader("Decision Breakdown")
            decision_counts = st.session_state.bid_requests_df_decided_rtb['Decision'].value_counts() # Use unique key
            if not decision_counts.empty:
                 fig10, ax10 = plt.subplots(figsize=(6, 4))
                 decision_counts.reindex(["Bid", "Skip"], fill_value=0).plot(kind='bar', ax=ax10, color=['green', 'red'])
                 ax10.set_ylabel("Count")
                 ax10.set_title("Bid Decision Counts")
                 plt.xticks(rotation=0)
                 st.pyplot(fig10)
                 plt.close(fig10)
            else:
                 st.info("No bid decisions to display.")


            skip_reasons = st.session_state.bid_requests_df_decided_rtb[st.session_state.bid_requests_df_decided_rtb['Decision'] == 'Skip']['Decision Reason'].value_counts() # Use unique key
            if not skip_reasons.empty:
                 st.subheader("Reasons for Skipping")
                 fig10_reasons, ax10_reasons = plt.subplots(figsize=(8, 4))
                 skip_reasons.plot(kind='bar', ax=ax10_reasons, color='salmon')
                 ax10_reasons.set_ylabel("Count")
                 ax10_reasons.set_title("Reasons for Skipping Bid")
                 plt.xticks(rotation=45, ha='right')
                 plt.tight_layout()
                 st.pyplot(fig10_reasons)
                 plt.close(fig10_reasons)
            elif 'Bid' in st.session_state.bid_requests_df_decided_rtb['Decision'].unique(): # Use unique key
                 st.info("No bids were skipped based on the current rules.")


    st.markdown("---")
    st.markdown("*Illustrates RTB, where quantum algorithms could provide millisecond speedups for complex bidding decisions in programmatic advertising.*")


# Update Sidebar About Section
with st.sidebar:
    st.title("🌀 About This App")
    st.markdown("""
    This quantum-inspired prototype demonstrates how quantum algorithms like Grover’s Search and Amplitude Amplification, along with quantum optimization and machine learning concepts, could potentially enhance modern retail and advertising decision-making.

    - 🔍 **Campaign Optimization**: Fast search of best ad combos using simulated search/optimization.
    - 🧠 **Sentiment Filtering**: Catch brand risks quickly using simulated threshold filtering.
    - 📋 **Project Tracker**: Monitor multiple projects with classical filtering and visualization.
    - 🎯 **Personalization Booster**: Enhance targeting with quantum-inspired probability boosting (Amplitude Amplification simulation).
    - 🚨 **Anomaly Detection**: Spot fraud and fake activity fast using statistical outlier detection (conceptually linkable to QPCA).
    - 📍 **Ad Placement Optimizer**: Optimize multi-screen ad plans using simulated combinatorial selection (relates to Quantum Annealing/QAOA).
    - 👥 **Audience Prediction Engine**: Forecast audience behavior with simulated data analysis (relates to Quantum ML).
    - 🚚 **Logistics & Scheduling**: Tackle physical deployment challenges efficiently with simulated task lists and sorting (relates to Quantum Routing/Optimization).
    - 🔮 **Forecast Simulator**: Model campaign outcomes with probabilistic simulations (relates to Quantum Simulations/Monte Carlo).
    - ⚡ **Real-Time Bidding (RTB)**: Quantum acceleration in programmatic DOOH bidding demonstrated via fast simulated decision rules.

    **Disclaimer:** This app uses *simulated* or *analogous* classical methods to illustrate the *concepts* of how quantum computing *might* be applied. It does **not** run actual quantum algorithms.

    Built using **Streamlit**.
    """)
    st.markdown("---")
    st.markdown("*Prototype by [Your Name/Organization - Optional]*")