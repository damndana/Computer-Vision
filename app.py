from dotenv import load_dotenv
load_dotenv() # load the environment variables

import streamlit as st 
import os
import pathlib
import textwrap
from PIL import Image
import google.generativeai as genai


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content([input, image[0], prompt])
    return response.text 


def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    
def main():
    
    st.set_page_config(page_title="Nutritionist-Food-Recognition-APP", page_icon="🍲")
    st.header("Your Dietitian and Nutritionist")
    
    language_options = ["English"]
    selected_language = st.selectbox("Select Language:", language_options)
    
    if selected_language == "English":
        input_prompt1 = """
        Embark on a culinary exploration as you uncover the secrets of the delectable dish captured in the uploaded image:
        1. Discover key details about the dish, including its name and culinary essence.
        2. Explore the fascinating origins of the dish, unraveling its cultural and historical significance.
        3. Dive into the rich tapestry of ingredients, presented pointwise, that contribute to the dish's exquisite flavor profile.
        """
        
        input_prompt2 = """
        As the culinary maestro guiding eager chefs, lay out the meticulous steps for crafting the featured dish:
        1. Start with selecting the finest ingredients, emphasizing quality and freshness.
        2. Detail the process of washing, peeling, and chopping each ingredient with precision.
        3. Unveil the culinary artistry behind the cooking process, step by step.
        4. Share expert tips and techniques to elevate the dish from ordinary to extraordinary.
        """
        
        input_prompt3 = """
        In your role as a nutritional advisor, present a comprehensive overview of the dish's nutritional value:
        1. Display a table showcasing nutritional values in descending order, covering calories, protein, fat, and carbohydrates.
        2. Create a second table illustrating the nutritional contribution of each ingredient, unraveling the dietary secrets within.
        """
        
        input_prompt4 = """
        Act as a dietitian and nutritionist:
        1. Your task is to provide 2 vegeterian dish alternative to the dish uploaded in the image which have the same nutritional value.
        2. Your task is to provide 2 Non-vegeterian dish alternative to the dish uploaded in the image which have the same nutritional value.
        """
        
    
        
    input_text = st.text_input("Input Prompt: ", key="input")
    
    uploaded_file = st.file_uploader("Choose an image ...", type=["jpg", "jpeg", "png"])
    image = ""
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        
        
    col1, col2 = st.columns(2)
    
    submit1 = col1.button("Get Dish Name and Ingredients")
    submit2 = col1.button("How to Cook")
    submit3 = col2.button("Nutritional Value for 1 Person")
    submit4 = col2.button("Alternative Dishes with Similar Nutritional Values")
    
    
    if submit1:
        if uploaded_file is not None:
            pdf_content = input_image_setup(uploaded_file)
            response = get_gemini_response(input_prompt1, pdf_content, input_text)
            st.subheader("The Response is")
            st.write(response)
        else:
            st.write("Please upload the dish image.")

    if submit2:
        if uploaded_file is not None:
            pdf_content = input_image_setup(uploaded_file)
            response = get_gemini_response(input_prompt2, pdf_content, input_text)
            st.subheader("The Response is")
            st.write(response)
        else:
            st.write("Please upload the dish image.")

    if submit3:
        if uploaded_file is not None:
            pdf_content = input_image_setup(uploaded_file)
            response = get_gemini_response(input_prompt3, pdf_content, input_text)
            st.subheader("The Response is")
            st.write(response)
        else:
            st.write("Please upload the dish image.")

    if submit4:
        if uploaded_file is not None:
            pdf_content = input_image_setup(uploaded_file)
            response = get_gemini_response(input_prompt4, pdf_content, input_text)
            st.subheader("The Response is")
            st.write(response)
        else:
            st.write("Please upload the dish image.")
    

if __name__ == "__main__":
    main()

