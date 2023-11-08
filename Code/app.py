import numpy as np 
import pandas as pd
import streamlit as st 
from function import visualize_timeseries ,yoy_growth,calculate_trend_slope_dataframe,generate_bedrock_response, recommend_products,generate_image,base64_to_pil,generate_bedrock_jurasic


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(
            page_title="Sigmoid GenAI",
            page_icon="Code/cropped-Sigmoid_logo_3x.png",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
st.sidebar.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
st.sidebar.image("Code/cropped-Sigmoid_logo_3x.png", use_column_width=True)
st.sidebar.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
st.markdown('<style>div.row-widget.stButton > button:first-child {background-color: blue; color: white;}</style>', unsafe_allow_html=True)

# def select_country(d):
#     country = st.sidebar.selectbox("Select Country:", d["geo"].unique().tolist())
#     return country

# def select_level(d):
#         levels = ["geo", "category", "brand", "SKU"]
#         selected_levels = st.sidebar.multiselect("Select Levels", levels, default=["geo"])

#         selected_category = None
#         selected_brand = None
#         selected_SKU = None

#         if "category" in selected_levels:
#             st.sidebar.header("Category")
#             category_options = d["category"].unique().tolist()
#             selected_category = st.sidebar.selectbox("Select category:", category_options)

#         if "brand" in selected_levels:
#             st.sidebar.header("Brand")
#             brand_options = d["brand"].unique().tolist()
#             selected_brand = st.sidebar.selectbox("Select brand:", brand_options)

#         if "SKU" in selected_levels:
#             st.sidebar.header("SKU")
#             SKU_options = d["SKU"].unique().tolist()
#             selected_SKU = st.sidebar.selectbox("Select SKU:", SKU_options)

#         return selected_levels, selected_category, selected_brand, selected_SKU

df_dash = pd.read_csv("Data/retail3.csv")
tab1, tab2 ,tab3,tab4,tab5,tab6= st.tabs(["About the App", "Demand forecasting interpreater","CodeAI","Personlized Message","Image Gen","Q&A"])
with tab1:
        st.markdown('<p style="color:red; font-size:30px; font-weight:bold;">About The App:</p>', unsafe_allow_html=True)
        st.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)   
        st.markdown("👋 Welcome to Sigmoid GenAI - Your Data Analysis APP!")
        st.write("This app is designed to help you analyze and visualize your data.")
        st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">👨‍💻  How to Use:</p>', unsafe_allow_html=True)
        st.write("1. From the top this page please select the required tab")
        st.write("2. Follow the instruction of that tab.")
        st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
        st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">Limitations ⚠️:</p>', unsafe_allow_html=True)
        st.write("Please note the following limitations:")
        st.write("- Active internet connection is required.")
        st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
with tab2:
    def main():
                def select_level(d):
                    """
                    Select data levels and additional options.

                    Parameters:
                    - d: DataFrame containing the data.

                    Returns:
                    - A tuple containing selected levels and other options.
                    """
                    
                    

                    # Create a list to store selected options
                    selected_levels = []
                    col_cou1,col_cou2,col_cou3,col_cou4=st.columns(4)
                    with col_cou1:
                        geo_options = d["geo"].unique().tolist()
                        st.markdown('<p style="border: 2px solid red; padding: 1px; font-weight: bold;color: blue;size:4;">Select Country:</p>', unsafe_allow_html=True)
                        selected_geo = st.selectbox("", geo_options)
                        d=d[d["geo"]==selected_geo]
                        selected_levels.append("geo")
                    c1,c2,c3,c4=st.columns(4)

                    with c1:
                        st.markdown('<p style="border: 2px solid red; padding: 0.1px; font-weight: bold;color: blue;">Select Hierarchy :</p>', unsafe_allow_html=True)
                    # Create columns for checkboxes
                    col1, col2, col3 = st.columns(3)
                    
                    #Create a checkbox for each level
                    with col1:
                        #st.markdown('<span style="font-size: 20px;"><font color="blue" size=4><b>category:</b></font></span>', unsafe_allow_html=True)
                        checkbox = st.checkbox("###### :red[category] 🛒", value="category" in selected_levels, key="category")
                        if checkbox:
                            selected_levels.append("category")

                    with col2:
                        #st.markdown('<span style="font-size: 20px;"><font color="blue" size=4><b>brand:</b></font></span>', unsafe_allow_html=True)
                        checkbox = st.checkbox("###### :red[brand] 🍺", value="brand" in selected_levels, key="brand")
                        if checkbox:
                            selected_levels.append("brand")

                    with col3:
                        #st.markdown('<span style="font-size: 20px;"><font color="blue" size=4><b>SKU:</b></font></span>', unsafe_allow_html=True)
                        checkbox = st.checkbox("###### :red[SKU] 💲", value="SKU" in selected_levels, key="SKU")
                        if checkbox:
                            selected_levels.append("SKU")
                    #selected_geo="Great Britain"
                    selected_category = None
                    selected_brand = None
                    selected_SKU = None
                    # Create columns for select boxes
                    col1, col2, col3= st.columns(3)
                    with col1:
                        if "category" in selected_levels:
                            category_options = d["category"].unique().tolist()
                            #st.markdown('<p style="border: 2px solid red; padding: 1px; font-weight: bold;color: blue;size:4;">Select category:</p>', unsafe_allow_html=True)
                            selected_category = st.selectbox("", category_options)
                            d=d[d["category"]==selected_category]

                    with col2:
                        if "brand" in selected_levels:
                            brand_options = d["brand"].unique().tolist()
                            #st.markdown('<p style="border: 2px solid red; padding: 1px; font-weight: bold;color: blue;size:4;">Select brand:</p>', unsafe_allow_html=True)
                            selected_brand = st.selectbox("", brand_options)
                            d=d[d["brand"]==selected_brand]

                    with col3:
                        if "SKU" in selected_levels:
                            SKU_options = d["SKU"].unique().tolist()
                            #st.markdown('<p style="border: 2px solid red; padding: 1px; font-weight: bold;color: blue;size:4;">Select SKU:</p>', unsafe_allow_html=True)
                            selected_SKU = st.selectbox("", SKU_options)

                    return selected_levels, selected_category, selected_brand, selected_SKU,selected_geo


            
                selected_levels = select_level(df_dash)

                # Time Series Visualization Section
                st.subheader("Visualize your time series")
                st.markdown("---")
                data = visualize_timeseries(df_dash,selected_levels[0], selected_levels[4],
                                            selected_levels[1], selected_levels[2], selected_levels[3])    
                data_trend = calculate_trend_slope_dataframe(data)
                if data_trend.empty:
                    pass
                else:
                    data_trend_2=data_trend.groupby(["scenario","trend"])[["slope_his","slope_for"]].mean().reset_index()
                if data.empty:
                    pass
                else:
                    data_yoy = yoy_growth(data)
                data_trend_3 =data_trend_2[["scenario","trend"]]
                # st.write(data_trend_3)
                # st.write(data_yoy.set_index('year')["yoy_growth"].to_dict())
                if st.button("Get Analysis"):
                    instruction = """
                    You are functioning as an AI data analyst.\
                    You will be analyzing two datasets: trend_dataset and year-on-year growth dataset.\
                    Trend_dataset has the following columns:\
                        - Scenario: Indicates if a data point is historical or forecasted.\
                        - Trend: Indicates the trend of the data for a specific scenario.\
                    Year on year growth dataset has the following columns:\
                        - Year: Indicates the year.\
                    - yoy_growth: Indicates the percentage quantity change compared to the previous year.\
                    Start the output as 'Insight and Findings:' and report the findings in point form. \
                    Analyze the trend based on trend_dataset: \
                    Analyze the year on year growth based on the year-on-year growth dataset, also include the change percentage.\
                    Use at most 200 words. \
                    Provide conclusions about the dataset's performance in 50 words over the year\
                    Give the report as it look like it is genrated by human

                    """
                    prompt = f"""
                    Analyse the provided trend dataset and year on year growth dataset which are delimited with html tags\
                    Follow the instruction which is delimited with triple backticks

                    Instruction: '''{instruction}'''
                    Trend Dataset: <{data_trend_3}/>
                    Year on Year Growth dataset : <{data_yoy}/>
                    """
                    st.write(generate_bedrock_response(prompt))

            

                # st.write(generate_bedrock_response(prompt))
                
    if __name__ == "__main__":
        main()
with tab3:



    # Initialize an empty dictionary to store column descriptions
        column_descriptions = {}

        def main():
            st.markdown('<p style="color:red; font-size:30px; font-weight:bold;">CodeAI:</p>', unsafe_allow_html=True)
            st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
            st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">👨‍💻 How to Use:</p>', unsafe_allow_html=True)
            st.markdown("""
            - 📂 Upload a CSV or Excel file containing your dataset.
            - 📝 Provide descriptions for each column of the dataset in the 'Column Descriptions' section.
            - 📂 Optionally, upload a CSV or Excel file containing column descriptions.
            - ❓ Ask questions about the dataset in the 'Ask a question about the dataset' section.
            - 🔍 Click the 'Get Answer' button to generate an answer based on your question.
            """)

            # Display limitations with emojis
            st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
            st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">Limitations ⚠️:</p>', unsafe_allow_html=True)
            st.markdown("""
            - The quality of AI responses depends on the quality and relevance of your questions.
            - Ensure that you have a good understanding of the dataset columns to ask relevant questions.
            """)   
            st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)

            # Upload the dataset file
            uploaded_file = st.file_uploader("Upload a CSV or Excel file (Dataset)", type=["csv", "xlsx"])
            st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">Head of the Dataset:</p>', unsafe_allow_html=True)
            
            df_user = pd.DataFrame()
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df_user = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                        df_user = pd.read_excel(uploaded_file)
                    
                    # Display the first few rows of the dataset
                    st.write(df_user.head())
                    

                    st.info("Please add column descriptions of your dataset")
                    for col in df_user.columns:
                        col_description = st.text_input(f"Description for column '{col}':")
                        if col_description:
                            column_descriptions[col] = col_description
                        
                    if st.button("Submit Descriptions"):
                        st.success("Descriptions submitted successfully!")
                except Exception as e:
                    st.error(f"An error occurred while reading the dataset file: {e}")
                    return

            # Optionally, upload column descriptions file
            st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
            uploaded_desc_file = st.file_uploader("Upload a CSV or Excel file (Column Descriptions)", type=["csv", "xlsx"])
            if uploaded_desc_file is not None:
                try:
                    if uploaded_desc_file.name.endswith('.csv'):
                        desc_df = pd.read_csv(uploaded_desc_file)
                    elif uploaded_desc_file.name.endswith(('.xls', '.xlsx')):
                        desc_df = pd.read_excel(uploaded_desc_file)

                    # Assuming the column descriptions are in two columns: 'Column Name' and 'Description'
                    for index, row in desc_df.iterrows():
                        col_name = row['Column Name']
                        col_description = row['Description']
                        if col_name and col_description:
                            column_descriptions[col_name] = col_description

                    st.success("Column descriptions loaded successfully!")
                except Exception as e:
                    st.error(f"An error occurred while reading the column descriptions file: {e}")

        


            
            st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
            st.markdown('<p style="color:red; font-size:25px; font-weight:bold;">Ask a question about the dataset:</p>', unsafe_allow_html=True)
            instruction = """
            You are acting as an AI data analyst. Your task is to respond to questions based on the provided dataset columns and their descriptions by providing executable code. Follow these instructions:

            1. Task: Respond to questions based on the dataset columns and descriptions by generating code.
            2. Columns Description: Use the provided dictionary format for column descriptions - {column_descriptions}.
            3. Provide code based on the user's question.
            4. Use the DataFrame 'df_user' - do not create any dummy datasets.
            5. Display the result using 'st.write' for text or 'st.pyplot' for plots, using Plotly with a white background.
            6. Return the code only - no explanations or additional text.
            7. Include code to suppress warnings.
            8. Do not include [assistant].
            9. Do not read any dataset; call the function with 'df_user'.
            10. Return the final output with 'st.write' or 'st.pyplot'.
            11. The code must start with 'def' and end with the function call.
            12. Do not include code in backticks.
            13. Provide only the executable code - no non-executable characters.
            14. Call the function below the response in the same script with 'df_user' as input.
            15.Ensure the code is syntactically correct, bug-free, optimized, not span multiple lines unnessarily, and prefer to use standard libraries. Return only python code without any surrounding text, explanation or context.

            """
            user_question = st.text_input(" ")
            user_question+="Return the final output of the function with 'st.write' or 'st.pyplot' and Call the function below"
            prompt = f"""
            Generate the code based on the user question, which is delimited with triple backticks, based on the dataframe columns and their descriptions, which are delimited with HTML tags.
            Follow the instruction, which is delimited with triple backticks.
            Instruction: '''{instruction}'''
            Columns: <{df_user.columns.tolist()}/>
            Columns description: <{column_descriptions}/>
            User question: '''{user_question}'''
            """



            if st.button("Get Answer"):
                if user_question:
                    user_message = generate_bedrock_response(prompt)
                    st.write(user_message)
                    # code = chat2(user_message)
                    # st.code(code.content)
                    # exec(user_message)
                else:
                    st.warning("Not a valid question. Please enter a question to analyze.")
            
            st.markdown('<p style="color:red; font-size:25px; font-weight:bold;">Code Execution Dashboard:</p>', unsafe_allow_html=True)
        
            st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
            code_input = st.text_area("Enter your code here", height=200)
            st.warning(("⚠️ If there is any non-executable line in generated code; please remove it"))
            
            if st.button("Execute code"): 
                try:
                    # Use exec() to execute the code
                    exec(code_input)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        # Check if the script is run as the main program
        if __name__ == "__main__":
            main()
with tab4:
    df_final = pd.read_csv("Data/df_final_with_name2.csv")
    existing_user = df_final["user_id"].unique()
    customer_style = """British English \
            thrilled and delighted
            """


    instruction_existing= """
                    1. Create a captivating message within an 80-word limit for maximum impact.
                    2. Add relevant emojis and apply formatting to enhance the message's appeal.
                    3. Maintain clear structure with suitable line breaks.
                    5. Focus solely on the message content; avoid extra information.
                    6. Ensure the message grabs attention.
                    7. Limit consecutive lines to a maximum of two for readability.
                    8. Format the message crisply for an attention-grabbing effect.
                    9. Use appropriate spacing between lines to increase attractiveness.
                    10. Avoid more than two consecutive sentences in the message toghter use proper space in between sentences.
                    """





    best_selling_product= """
    The best selling products:-
    1.Ciroc Vodka
    2.Black & White Blended Whisky
    """


    welcome_offer="""1.Free express shipping for a limited time.
                        2.Give $10, get $10 when you refer a friend."""
    # Function to generate personalized messages for new users
    def personlized_message_new_user(style, welcome_offer, best_selling_pro, user_data,instruction_existing):

        prompt =f"""Generate a personalized welcome message for new users as they log in to the 'Diageo' website. 
                            Utilize user data ({user_data}) to tailor the message must include name. 
                            Highlight the best-selling products ({best_selling_pro}) 
                            and current welcome offers ({welcome_offer}). 
                            Present the message in a stylish format that is {style}
                            Please adhere to the provided instructions: {instruction_existing}.
                            """

        mesage= generate_bedrock_response(prompt)
        return mesage

    # Function to generate personalized messages for existing users
    def personlized_message_existing_user(style, Existing_user_data, Rec_product, Offers_and_promotion,instruction_existing):
        prompt_user = f"""Generate a personalized welcome message for users logging into the Diageo website.
                        Utilize existing user data, including {Existing_user_data} include the name in begning, 
                        to provide tailored product recommendations ({Rec_product}). 
                        Incorporate cart item promotions and offers ({Offers_and_promotion}). 
                        Present the message in a stylish format ({style}). 
                        Follow the provided instructions carefully: {instruction_existing}.
                    """
        message= generate_bedrock_response(prompt_user)
        return message

    # Define custom colors
    primary_color = "#3498db"  # Blue
    secondary_color = "#2ecc71"  # Green
    background_color = "#f0f3f6"  # Light Gray
    text_color = "#333333"  # Dark Gray

    # Apply custom styles
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: {background_color};
            color: {text_color};
        }}
        .sidebar .sidebar-content {{
            background: {primary_color};
            color: white;
        }}
        .widget-label {{
            color: {text_color};
        }}
        .stButton.button-primary {{
            background: {secondary_color};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Recommendation part
    st.markdown("### 🎁 Personalized Welcome Message")
    st.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
    # st.subheader("Login to get personalized recommendations.")

    with st.form("login_form"):
        user_id = st.text_input("User ID")
        user_name = st.text_input("Your Name")
        submitted = st.form_submit_button("Login")

    if submitted:
        if user_id in existing_user:
            if user_id:
                recommended_products = recommend_products(user_id, df_final)

                if recommended_products:
                    offer = df_final[df_final["product_id"].isin(recommended_products)]["offers"].unique().tolist()
                    offers = [offer[0], offer[-1]]
                    Existing_user_data = {"Name": user_name, "Existing Items in the cart": ["Tanqueray Sterling Vodka", "7 Crown Appl", "Ursus Punch Vodka"]}
                    Rec_product = recommended_products
                    Offers_and_promotion = offers
                    existing_user = personlized_message_existing_user(customer_style, Existing_user_data, Rec_product, Offers_and_promotion,instruction_existing)
                    with st.chat_message("user"):
                        st.write(existing_user)
                else:
                    st.warning("No recommendations available for this user.")
        else:
            new_message = personlized_message_new_user( customer_style, welcome_offer, best_selling_product, user_name,instruction_existing)
            with st.chat_message("user"):
                st.write(new_message)
with tab5:
    # List of Stable Diffusion Preset Styles
    sd_presets = ["None","3d-model","analog-film","anime","cinematic","digital-art","photographic"]

    # select box for styles
    style = st.selectbox("Select Style", sd_presets)
    # text input
    prompt = st.text_input("Enter prompt")

    #  Generate image from prompt,
    if st.button("Generate Image"):
        image = base64_to_pil(generate_image(prompt, style))
        st.image(image)
# with tab6:
#     st.markdown('<p style="color:red; font-size:30px; font-weight:bold;">DocAI:</p>', unsafe_allow_html=True)
#     st.markdown("<hr style='border: 2px solid red; width: 100%;'>", unsafe_allow_html=True)
#     st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">👨‍💻  How to Use:</p>', unsafe_allow_html=True)
#     st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
#     st.markdown('1. **Upload an Article**: Click on the "Upload an article" button to upload a text (.txt) or PDF (.pdf) file containing the content you want to query.')
#     st.markdown('2. **Enter your Question**: Enter your question or query in the "Enter your question" field. This question will be used to generate a response based on the uploaded content.')
#     st.markdown('3. **Generate Response**: After uploading the file and entering the question, click the "Generate Response" button. This will trigger the response generation process.')
#     st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
#     # Limitations
#     st.markdown('<p style="color:blue; font-size:20px; font-weight:bold;">Limitations ⚠️:</p>', unsafe_allow_html=True)
#     st.markdown('1. **Supported File Formats**: Only text (.txt) and PDF (.pdf) file formats are supported for uploading. Other formats are not supported.')
#     st.markdown('2. **Query Text Required**: You must enter a question or query in the "Enter your question" field. Without a question, you cannot generate a response.')
#     st.markdown('3. **Response Time**: The response generation process may take some time, depending on the complexity of the query and the size of the uploaded file.')
#     st.markdown("<hr style='border: 1.5px solid red; width: 100%;'>", unsafe_allow_html=True)
#     def generate_response(uploaded_file, openai_api_key, query_text):
#             # Load document if file is uploaded
#         if uploaded_file is not None:
#             documents = [uploaded_file.read().decode()]
#                 # Split documents into chunks
#             text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#             texts = text_splitter.create_documents(documents)
#                 # Select embeddings
#             embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#                 # Create a vectorstore from documents
#             db = FAISS.from_documents(texts, embeddings)
#                 # Create retriever interface
#             retriever = db.as_retriever()
#                 # Create QA chain
#             qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
#             return qa.run(query_text)
#     def pdf_chat(uploaded_file,query):
#         if uploaded_file is not None:
#             pdf_reader = PdfReader(uploaded_file)

#             text = ""
#             for page in pdf_reader.pages:
#                 text+= page.extract_text()

#                 #langchain_textspliter
#             text_splitter = RecursiveCharacterTextSplitter(
#                     chunk_size = 1000,
#                     chunk_overlap = 200,
#                     length_function = len
#                 )

#             chunks = text_splitter.split_text(text=text)

                
#                 #store pdf name
#             store_name = uploaded_file.name[:-4]
                
#             if os.path.exists(f"{store_name}.pkl"):
#                 with open(f"{store_name}.pkl","rb") as f:
#                     vectorstore = pickle.load(f)
#                     #st.write("Already, Embeddings loaded from the your folder (disks)")
#             else:
#                     #embedding (Openai methods) 
#                 embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

#                     #Store the chunks part in db (vector)
#                 vectorstore = FAISS.from_texts(chunks,embedding=embeddings)

#                 with open(f"{store_name}.pkl","wb") as f:
#                     pickle.dump(vectorstore,f)
                    
#                     #st.write("Embedding computation completed")

#                 #st.write(chunks)
                
#                 #Accept user questions/query

        
#                 #st.write(query)

#             if query:
#                 docs = vectorstore.similarity_search(query=query,k=3)
#                     #st.write(docs)
                    
#                     #openai rank lnv process
#                 llm = Bedrock(
#                                 credentials_profile_name="bedrock-admin",
#                                 model_id="cohere.command-text-v14"
#                             )
                    
#                 chain = load_qa_chain(llm=llm, chain_type= "stuff")
                    
#                 with get_openai_callback() as cb:
#                     response = chain.run(input_documents = docs, question = query)
                        
#         return response
#     def main():
#             # File upload
#         uploaded_file = st.file_uploader('Upload an article', type=['txt', 'pdf'])

#             # Query text
#         query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)
#         query_text+= "Always give the complete answer"

#             # Form input and query
#         result = []

#         if st.button('Generate Response'):  # Add a button to trigger response generation
#             with st.spinner('Generating...'):
#                 if query_text:
#                     if uploaded_file.type == 'text/plain':
#                         response = generate_response(uploaded_file, openai_api_key, query_text)
#                         result.append(response)
#                     elif uploaded_file.type == 'application/pdf':
#                             # Handle plain text file
#                         response = pdf_chat(uploaded_file,query_text)
#                         result.append(response)
#                     else:
#                         response = "Unsupported file format. Please upload a PDF or text file."


#         if len(result):
#             st.write(result[0])  # Display the response if there is a result

#     if __name__ == "__main__":
#         main()

            