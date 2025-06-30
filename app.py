import streamlit as st
from streamlit_option_menu import option_menu
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Set page configuration
st.set_page_config(
    page_title="VitalCare GPT",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Hugging Face models and tokenizers
@st.cache_resource
def load_models():
    pipe_disease = pipeline("text-generation", model="harishussain12/PastelMed")
    tokenizer_lynxmed = AutoTokenizer.from_pretrained("harishussain12/LynxMed")
    model_lynxmed = AutoModelForCausalLM.from_pretrained("harishussain12/LynxMed")

    tokenizer_neuramed = AutoTokenizer.from_pretrained("harishussain12/NeuraMed")
    model_neuramed = AutoModelForCausalLM.from_pretrained("harishussain12/NeuraMed")

    tokenizer_skyemed = AutoTokenizer.from_pretrained("harishussain12/SkyeMed")
    model_skyemed = AutoModelForCausalLM.from_pretrained("harishussain12/SkyeMed")

    tokenizer_clixmed = AutoTokenizer.from_pretrained("harishussain12/ClixMed")
    model_clixmed = AutoModelForCausalLM.from_pretrained("harishussain12/ClixMed")

    return pipe_disease, (tokenizer_lynxmed, model_lynxmed), (tokenizer_neuramed, model_neuramed), (tokenizer_skyemed, model_skyemed), (tokenizer_clixmed, model_clixmed)

# Function to create pipelines for all models
@st.cache_resource
def create_pipelines():
    pipe_disease, (tokenizer_lynxmed, model_lynxmed), (tokenizer_neuramed, model_neuramed), (tokenizer_skyemed, model_skyemed), (tokenizer_clixmed, model_clixmed) = load_models()
    pipeline_lynxmed = pipeline("text-generation", model=model_lynxmed, tokenizer=tokenizer_lynxmed)
    pipe_neuramed = pipeline("text-generation", model=model_neuramed, tokenizer=tokenizer_neuramed)
    pipe_skyemed = pipeline("text-generation", model=model_skyemed, tokenizer=tokenizer_skyemed)
    pipeline_clixmed = pipeline("text-generation", model=model_clixmed, tokenizer=tokenizer_clixmed)

    return {
        "PastelMed": pipe_disease,
        "LynxMed": pipeline_lynxmed,
        "NeuraMed": pipe_neuramed,
        "SkyeMed": pipe_skyemed,
        "ClixMed": pipeline_clixmed
    }

# Load pipelines
pipelines = create_pipelines()

# Sidebar with navigation
with st.sidebar:
    selected = option_menu(
        menu_title=None,  # Remove the Navigation title
        options=["Home", "Spaces", "About"],
        icons=["house", "search", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#3e4a5b"},
            "icon": {"color": "#ffffff", "font-size": "16px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "0px",
                "color": "#ffffff",
                "font-weight": "bold",
                "padding": "10px 20px",
            },
            "nav-link-selected": {"background-color": "#0b2545", "color": "white"},
        }
    )

# Initialize session state for chat history
if 'home_chat_history' not in st.session_state:
    st.session_state['home_chat_history'] = []

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = {}

# Define role-specific keywords or categories
role_keywords = {
    "Doctor": ["symptoms", "diagnosis", "treatment", "disease", "medical condition", 
               "prescription", "procedure", "surgery", "consultation", "therapy", 
               "prognosis", "clinical", "specialist", "check-up", "imaging", 
               "laboratory tests", "pathology", "epidemiology", "anatomy", "physiology"],
    "Nutritionist": ["diet", "nutrition", "meal plan", "calories", "weight management", 
                     "vitamins", "minerals", "protein", "carbohydrates", "fats", 
                     "healthy eating", "lifestyle", "food", "allergies", "deficiencies", 
                     "hydration", "superfoods", "balanced diet", "supplements", "recipes"],
    "Pharmacist": ["medication", "dosage", "side effects", "drug", "pharmacy", 
                   "prescription", "over-the-counter", "interaction", "refill", 
                   "formulation", "pharmacology", "pharmaceutical", "compounding", 
                   "instructions", "contraindications", "storage", "expiry", 
                   "dispense", "pharmacist advice", "generic drugs", "medicine"]
}

# Role prompts
role_prompts = {
    "Doctor": """
        You are assisting as a doctor.
        Tasks:
        - Answer medical questions concisely and accurately.
        - Respond with: "I don't know about it" if the query is not related to the medical field.
    """,
    "Nutritionist": """
        You are assisting as a nutritionist.
        Tasks:
        - Provide dietary advice based on queries.
        - Suggest meal plans, calorie intake, and balanced diets.
        - Respond with: "I don't know about it" if the query is not related to nutrition.
    """,
    "Pharmacist": """
        You act as a pharmacist.
        Tasks:
        - Provide details on medications, dosages, and side effects.
        - Respond with: "I don't know about it" if unrelated to medicine.
    """
}

# Function to check if query matches the role
def is_query_relevant(role, query):
    keywords = role_keywords.get(role, [])
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in keywords)

# Main content based on navigation
if selected == "Home":
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("<h1 style='text-align: center;'>VitalCare GPT</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>How can I assist with your medical queries today?</h3>", unsafe_allow_html=True)

        # Display chat history for Home section above input
        for message in st.session_state['home_chat_history']:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Model selection
        model_selection = st.selectbox(
            "Select a model",
            options=["SkyeMed", "NeuraMed", "PastelMed", "LynxMed", "ClixMed"],
            index=0
        )

        # Search box
        search_input = st.text_input(
            "",
            placeholder="Type your medical question here...",
            label_visibility="collapsed",
            help="Ask anything related to medical knowledge."
        )

        if search_input:
            with st.spinner("Generating response..."):
                try:
                    query_input = search_input
                    response = pipelines[model_selection](query_input, max_length=200, num_return_sequences=1)

                    # Save the user and assistant messages to chat history
                    st.session_state['home_chat_history'].append({"role": "user", "content": search_input})
                    st.session_state['home_chat_history'].append({"role": "assistant", "content": response[0]['generated_text']})

                    # Display the generated response
                    st.markdown(f"### Response:\n{response[0]['generated_text']}")

                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

elif selected == "Spaces":
    st.markdown("<h1>Spaces</h1>", unsafe_allow_html=True)

    # Layout for space buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Doctor", key="doctor", use_container_width=True):
            st.session_state.selected_role = "Doctor"
    with col2:
        if st.button("Nutritionist", key="nutritionist", use_container_width=True):
            st.session_state.selected_role = "Nutritionist"
    with col3:
        if st.button("Pharmacist", key="pharmacist", use_container_width=True):
            st.session_state.selected_role = "Pharmacist"

    # Display the selected role
    if "selected_role" in st.session_state:
        selected_role = st.session_state.selected_role
        st.markdown(f"<h2>Selected Space: {selected_role}</h2>", unsafe_allow_html=True)

        # Initialize chat history for the selected role if not already done
        if selected_role not in st.session_state['chat_history']:
            st.session_state['chat_history'][selected_role] = []

        # Display chat history for the selected role
        for message in st.session_state['chat_history'][selected_role]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Add model selection dropdown
        model_selection = st.selectbox(
            "Select a model",
            options=["SkyeMed", "NeuraMed", "PastelMed", "LynxMed", "ClixMed"],
            index=0
        )

        # Align query input and button on the same line
        query_col1, query_col2 = st.columns([4, 1])
        with query_col1:
            query = st.text_input(
                f"Enter your query as a {selected_role.lower()}:",
                placeholder="Type your question here...",
                label_visibility="collapsed"
            )
        with query_col2:
            generate_clicked = st.button("Generate Response", key="generate_button")

        if generate_clicked:
            if query.strip():
                with st.spinner("Generating response..."):
                    try:
                        # Check query relevance for the selected role
                        if not is_query_relevant(selected_role, query):
                            response_text = f"As a {selected_role.lower()}, I cannot answer this question."
                        else:
                            # Generate response using the selected model
                            role_prompt = role_prompts.get(selected_role, "")
                            formatted_query = f"\n\nquery: {query}\n"
                            response = pipelines[model_selection](formatted_query, max_length=200, num_return_sequences=1)
                            response_text = response[0]['generated_text']

                        # Save user and assistant messages to the selected role's chat history
                        st.session_state['chat_history'][selected_role].append({"role": "user", "content": query})
                        st.session_state['chat_history'][selected_role].append({"role": "assistant", "content": response_text})

                        # Display the response
                        st.markdown(f"### Response:\n{response_text}")

                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
            else:
                st.warning("Please enter a query before generating a response.")

elif selected == "About":
    st.markdown("<h1>About VitalCare GPT</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        VitalCare GPT is an advanced AI-powered platform designed to provide reliable and accurate medical insights, enabling users to access information related to healthcare and wellness effortlessly. Powered by cutting-edge language models, VitalCare GPT specializes in various domains, including general medical advice, nutritional guidance, and pharmaceutical expertise.
Whether you're looking for symptoms analysis, dietary recommendations, or medication details, our platform empowers users to interact seamlessly with AI models trained on specific medical and healthcare-related datasets. VitalCare GPT offers dedicated spaces for doctors, nutritionists, and pharmacists, ensuring tailored responses to your queries.
        """
    )

# Footer at the bottom with centered text, and adjusted when sidebar is toggled
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            padding: 8px;  /* Reduced padding */
            border-radius: 10px;
            font-size: 12px;  /* Smaller font size */
            text-align: center;
            z-index: 1000;
            background-color: transparent;
        }
        /* Adjust position based on sidebar */
        .footer-container {
            display: flex;
            justify-content: center;
            align-items: center;
            position: fixed;
            bottom: 10px;
            left: 50%;
            transform: translateX(-50%);
            width: 100%;
        }
        @media screen and (max-width: 900px) {
            .footer {
                position: fixed;
                left: 50%;
                transform: translateX(-50%);
            }
        }
    </style>
    <div class="footer-container">
        <div class="footer">
            This GPT may take time to generate responses and may have lower accuracy.
        </div>
    </div>
""", unsafe_allow_html=True)

# Floating question mark icon with tooltip
st.markdown("""
    <style>
        /* Floating Question Mark Icon */
        .help-icon {
            position: fixed;
            bottom: 10px;
            right: 10px;
            background-color: #333;
            color: white;
            font-size: 14px;  /* Even smaller font size */
            border-radius: 50%;
            padding: 6px;  /* Smaller padding */
            width: 30px;  /* Smaller width */
            height: 30px;  /* Smaller height */
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            z-index: 1000;
        }
        /* Tooltip content when hovering */
        .help-tooltip {
            position: fixed;
            bottom: 50px;
            right: 10px;
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 6px;  /* Smaller padding */
            border-radius: 10px;
            font-size: 12px;  /* Smaller font size */
            display: none;
            z-index: 1000;
        }
        .help-icon:hover + .help-tooltip,
        .help-tooltip:hover {
            display: block;
        }
    </style>
    <!-- Help icon and tooltip -->
    <div class="help-icon">?</div>
    <div class="help-tooltip">
        Developed by<br>
        Rayyan & Haris
    </div>
""", unsafe_allow_html=True)
