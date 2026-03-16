import streamlit as st
import pandas as pd
import joblib
import re
import time
import json
import google.generativeai as genai
from dotenv import load_dotenv
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Smart Car Advisor", page_icon="🚗", layout="wide")

load_dotenv()
# --- GEMINI CONFIGURATION ---
# ⚠️ PASTE YOUR API KEY IN FILE NAMED .env UNDER ROOT DIRECTORY ⚠️
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Define the model globally to ensure consistency
    model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
    chat_available = True
except Exception as e:
    chat_available = False

# --- 1. LOAD MODELS ---
@st.cache_resource
def load_models():
    try:
        price_model = joblib.load('models/price_model_optimized.pkl') 
        inspector_model = joblib.load('models/inspector_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return price_model, inspector_model, vectorizer
    except Exception:
        return None, None, None

price_model, inspector_model, vectorizer = load_models()

# --- 2. AI RECOMMENDER ENGINE ---
@st.cache_data(show_spinner=False)
def get_ai_recommendations(car_title, user_desc):
    """
    Uses Gemini to analyze the specific car and description to suggest
    high-ROI upgrades dynamically. Returns a JSON list.
    """
    if not chat_available:
        return []

    try:
        # 1. Strict Prompt to force JSON output
        prompt = f"""
        Act as a Pakistani Used Car Dealer Expert. 
        I am selling a: {car_title}
        Current Condition/Description: "{user_desc}"
        
        Identify 3 specific, high-return physical upgrades or fixes that would maximize the resale value of this SPECIFIC car in the Pakistani market.
        
        Rules:
        1. Do NOT suggest things already mentioned in the description.
        2. Be specific (e.g., instead of "Sound system", say "Android Panel").
        3. For each item, give a "Lift Score" (1-10) of how much it helps the price.
        4. Return ONLY a valid JSON list like this example:
        [
          {{"mod": "Android Panel", "reason": "High demand in Family cars", "lift": 9}},
          {{"mod": "Detailing/Compound", "reason": "Removes scratches", "lift": 7}}
        ]
        """
        
        # 2. Call Gemini
        response = model.generate_content(prompt)
        
        # 3. Clean and Parse JSON (Strip markdown if Gemini adds it)
        text = response.text.replace("```json", "").replace("```", "").strip()
        recommendations = json.loads(text)
        
        return recommendations
        
    except Exception as e:
        # Fallback if AI fails (Safety net)
        return [
            {"mod": "Detailing & Polishing", "reason": "Universal value booster", "lift": 5},
            {"mod": "High-Quality Pictures", "reason": "Increases click-through rate", "lift": 8}
        ]

# --- 3. HELPER FUNCTIONS ---
@st.cache_data
def load_car_titles():
    try:
        df = pd.read_csv('../data/MASTER_CAR_DATASET.csv')
        titles = sorted(df['title_version'].astype(str).unique().tolist())
        return titles
    except:
        return ["Honda Civic Oriel 1.8 i-VTEC CVT 2020"]

car_options = load_car_titles()

def smart_condition_tagger(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    tags = []
    
    if re.search(r"(file).*(miss|lost|dup|fake)", text) or "ncp" in text: tags.append("TAG_FILE_ISSUE")
    elif re.search(r"(book|card|copy).*(dup|miss|fake)", text): tags.append("TAG_BOOK_ISSUE")
    if re.search(r"(roof|pillar).*(paint|damage|shower)", text): tags.append("TAG_MAJOR_ACCIDENT")
    if re.search(r"(engine|gear).*(change|swap|replace|noise|smoke)", text): tags.append("TAG_MECHANICAL_BAD")
    if "shower" in text or "repaint" in text or "fresh look" in text: tags.append("TAG_COSMETIC_BAD")
    minor_pattern = r"\b(1|one|2|two|3|three)(\s+(or|to|-)\s+(2|two|3|three))?\s*(piece|pc|touch)"
    if re.search(minor_pattern, text) or "minor" in text: tags.append("TAG_MINOR")

    bad_tags = ["TAG_FILE_ISSUE", "TAG_BOOK_ISSUE", "TAG_MAJOR_ACCIDENT", "TAG_MECHANICAL_BAD", "TAG_COSMETIC_BAD", "TAG_MINOR"]
    if not any(t in tags for t in bad_tags):
        if "genuine" in text or "bumper" in text or "no touch" in text or "original" in text:
            tags.append("TAG_EXCELLENT")

    return text + " " + " ".join(list(set(tags)))

# --- MAIN APP LAYOUT ---

# 1. TOP NAVIGATION
app_mode = st.radio("Navigation", ["💰 Price Estimator", "🤖 AI Mechanic Chat"], horizontal=True, label_visibility="collapsed")

if app_mode == "💰 Price Estimator":
    # ==========================================
    # MODE 1: PRICE ESTIMATOR (Sidebar Visible)
    # ==========================================
    
    st.title("🚗 Smart Car Advisor")
    st.markdown("### Intelligent Price Estimation System")
    st.markdown("---")

    if price_model is None:
        st.error("🚨 Critical Error: Models not found.")
        st.stop()

    # --- SIDEBAR INPUTS ---
    st.sidebar.header("Vehicle Specifications")
    
    default_index = 0
    for i, title in enumerate(car_options):
        if "Civic Oriel 1.8 i-VTEC CVT 2020" in title:
            default_index = i
            break

    car_name = st.sidebar.selectbox("Select Car Model", options=car_options, index=default_index)
    year = st.sidebar.slider("Model Year", 2005, 2024, 2020)
    mileage = st.sidebar.number_input("Mileage (km)", 0, 300000, 50000, step=1000)
    engine = st.sidebar.number_input("Engine Capacity (cc)", 600, 6000, 1800, step=100)
    trans = st.sidebar.selectbox("Transmission", ["Automatic", "Manual"])
    fuel = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid"])

    # --- MAIN CONTENT ---
    st.subheader("📝 Seller Description Analysis")
    user_desc = st.text_area("Ad Description:", height=150, value="Total genuine bumper to bumper")

    if st.button("🚀 Calculate Market Value", type="primary"):
        tagged_desc = smart_condition_tagger(user_desc)
        condition_multiplier = 1.0
        status_msg = "Standard Market Condition"
        final_score = 9.0 
        
        if "TAG_FILE_ISSUE" in tagged_desc:
            condition_multiplier -= 0.35 
            status_msg = "⚠️ Major: Missing/Duplicate File"
            final_score = 5.0
        elif "TAG_BOOK_ISSUE" in tagged_desc:
            condition_multiplier -= 0.12 
            status_msg = "⚠️ Legal: Duplicate Book/Card"
            final_score = 7.0
        elif "TAG_MAJOR_ACCIDENT" in tagged_desc:
            condition_multiplier -= 0.25 
            status_msg = "❌ Structural Accident"
            final_score = 6.0
        elif "TAG_COSMETIC_BAD" in tagged_desc:
            condition_multiplier -= 0.10 
            status_msg = "🎨 Repainted / Showered"
            final_score = 7.5
        elif "TAG_MINOR" in tagged_desc:
            condition_multiplier -= 0.02 
            status_msg = "🖌️ Minor Touchups (1-2 Pieces)"
            final_score = 9.2
        elif "TAG_EXCELLENT" in tagged_desc:
            condition_multiplier += 0.03 
            status_msg = "💎 Bumper-to-Bumper Genuine"
            final_score = 9.8

        input_data = pd.DataFrame({
            'title_version': [car_name], 'model_year': [year], 'mileage': [mileage],
            'engine': [engine], 'transmission': [trans], 'fuel': [fuel],
            'inspection_score': [9.0] 
        })
        
        base_price = price_model.predict(input_data)[0]
        final_price = base_price * condition_multiplier

        st.markdown("---")
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.subheader("Condition Score")
            st.metric("AI Rating", f"{final_score:.1f} / 10")
            if "Genuine" in status_msg: st.success(status_msg)
            elif "Minor" in status_msg: st.info(status_msg)
            elif "Legal" in status_msg: st.warning(status_msg)
            else: st.error(status_msg)

        with col2:
            st.subheader("Market Valuation")
            st.metric("Estimated Price", f"PKR {int(final_price):,}")
            diff = final_price - base_price
            if diff > 0:
                st.caption(f"📈 Includes **PKR {int(diff):,} premium** for Genuine condition.")
            elif diff < 0:
                st.caption(f"📉 Includes **PKR {int(abs(diff)):,} discount** for condition/docs.")

        # --- NEW: AI-POWERED RECOMMENDER ---
        st.markdown("---")
        st.subheader("💡 AI Value Analyst")
        
        if chat_available:
            with st.spinner("🤖 AI is analyzing market gaps for your car..."):
                ai_suggestions = get_ai_recommendations(car_name, user_desc)
            
            if ai_suggestions:
                st.info(f"📈 **Strategy:** Based on the specific condition of your **{car_name}**, our AI identified these high-ROI opportunities:")
                
                cols = st.columns(3)
                # Handle cases where AI returns fewer than 3 items
                count = min(len(ai_suggestions), 3)
                for i in range(count):
                    item = ai_suggestions[i]
                    with cols[i]:
                        st.metric(
                            label=item['mod'], 
                            value=f"Score: {item['lift']}/10", 
                            delta="Value Booster",
                            help=item['reason']
                        )
                        st.caption(f"_{item['reason']}_")
        else:
            st.warning("⚠️ Connect Internet/API to see AI Recommendations.")

elif app_mode == "🤖 AI Mechanic Chat":
    # ==========================================
    # MODE 2: AI CHAT (Sidebar = Quick Actions)
    # ==========================================
    
    # 1. Define a variable to hold the user's input (from either source)
    user_prompt = None

    # --- SIDEBAR: QUICK ACTION BUTTONS ---
    with st.sidebar:
        st.header("🔧 Quick Actions")
        st.caption("Tap to ask instantly:")
        
        # If a button is clicked, we assign the text to 'user_prompt'
        if st.button("📈 Market Trends 2024"):
            user_prompt = "What are the current market trends for used cars in Pakistan in 2024?"
        
        if st.button("🛠️ Civic Maintenance"):
            user_prompt = "What is the maintenance schedule and cost for a Honda Civic X in Pakistan?"
            
        if st.button("🚙 Alto vs Mira"):
            user_prompt = "Compare Suzuki Alto 660cc vs Daihatsu Mira. Which has better resale and fuel average?"

        st.markdown("---")
        st.info("💡 **Tip:** I can compare prices, suggest mechanics, or estimate repair costs.")

    st.title("🤖 AI Mechanic Chat")
    st.caption("Powered by Gemini 2.5 Flash-Lite. Context-Aware & Concise.")

    if not chat_available:
        st.error("⚠️ Gemini API Key not found. Please add your key in the code.")
    else:
        # 2. Initialize Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your AI Mechanic. Ask me about maintenance, prices, or faults."}
            ]

        # 3. Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 4. Capture Chat Input (The Text Box)
        # We store this in a temporary variable
        chat_input_value = st.chat_input("Ask a question about your car...")

        # 5. PRIORITY LOGIC: Did they Type or Click?
        # If they typed something, use that. If not, check if they clicked a button.
        if chat_input_value:
            user_prompt = chat_input_value

        # 6. MAIN AI LOGIC (Runs if 'user_prompt' is not None)
        if user_prompt:
            # Display User Message
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Generate AI Response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    # --- MEMORY LOOP ---
                    history_text = ""
                    for msg in st.session_state.messages[-6:]:
                        role_label = "User" if msg["role"] == "user" else "AI"
                        history_text += f"{role_label}: {msg['content']}\n"

                    # --- SYSTEM INSTRUCTION ---
                    # FIX: We now use {user_prompt} instead of {prompt}
                    final_prompt = f"""
                    SYSTEM INSTRUCTION:
                    You are an expert car mechanic and market analyst for the Pakistani Used Car Market.
                    1. Keep your answers concise (200-250 words max).
                    2. Use bullet points for lists.
                    3. Use Pakistani currency (Lakhs, Crores) and terminology (Touching, Genuine, Shahnawaz Import, On-money).
                    4. Use the conversation history below to understand context.

                    CONVERSATION HISTORY:
                    {history_text}

                    NEW QUESTION:
                    {user_prompt}
                    """
                    
                    # Call Gemini
                    response = model.generate_content(final_prompt)
                    ai_text = response.text
                    
                    # Stream Output
                    for chunk in ai_text.split(): 
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    
                    # Save AI Response
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    # Detailed error for debugging
                    print(f"ERROR: {e}") 
                    st.error("⚠️ Network busy or Quota Exceeded. Please try again in 10 seconds.")