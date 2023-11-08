import numpy as np 
import pandas as pd
import plotly.express as px 
import streamlit as st
import numpy as np
import json
import boto3
from PIL import Image
import io
import base64

@st.cache_data
def visualize_timeseries(df, level, country, category, brand, SKU):
    df_t = df[df["geo"] == country]
    if category:
        df_t = df_t[df_t["category"] == category]
    if brand:
        df_t = df_t[df_t["brand"] == brand]
    if SKU:
        df_t = df_t[df_t["SKU"] == SKU]

    if df_t.empty:
        st.warning("No data available for the selected combination.")
    else:
        group_cols = level + ["month","scenario"]
        aggregation = {"quantity": "sum"}
        df_t = df_t.groupby(group_cols, as_index=True).agg(aggregation).reset_index()
    df_t = df_t.dropna()
    chart_data = df_t.set_index("month")
    title = "_".join([country] + [val for val in [category, brand, SKU] if val])
    color_discrete_map = {
        "actual": "blue",
        "predicted": "red"
    }

    quantity_chart = px.line(
        chart_data,
        x=chart_data.index,
        y="quantity",
        title=title,
        color="scenario",
        color_discrete_map=color_discrete_map
    )
    quantity_chart.update_layout(height=500, xaxis_title="Month", yaxis_title="Quantity")
    st.plotly_chart(quantity_chart, use_container_width=True)
    st.markdown("---")

    return df_t

@st.cache_data
def yoy_growth(df):
    df["year"] = pd.to_datetime(df["month"]).dt.year
    df_yoy = df.groupby(["year"]).sum()["quantity"].reset_index()
    grouped_yoy = df_yoy[2:-1]
    grouped_yoy['yoy_growth'] = grouped_yoy['quantity'].pct_change(periods=1) * 100
    return grouped_yoy[["year","yoy_growth"]]


@st.cache_data
def calculate_trend_slope_dataframe(dataframe, polynomial_degree=1):
    if dataframe.empty:
        st.warning("No data available for the selected combination.")
    else:
        dataframe=dataframe.reset_index(drop=True)
        df_copy = dataframe.copy() 
        df_copy['cumulative_sum'] = df_copy['quantity'].cumsum()
        first_nonzero_index = df_copy['cumulative_sum'].ne(0).idxmax()
        df_copy = df_copy.iloc[first_nonzero_index:]
        df_copy.drop(columns=['cumulative_sum'], inplace=True)
        df_copy_his =df_copy[df_copy["scenario"]=="actual"]
        df_copy_for = df_copy[df_copy["scenario"]=="predicted"]
        time_points_his = [i for i in range(len(df_copy_his["quantity"]))]
        quantity_values_his = df_copy_his["quantity"]
        coefficients_his = np.polyfit(time_points_his, quantity_values_his, polynomial_degree)
        slope_his = coefficients_his[0]
        df_copy_his["slope_his"]=slope_his
        if slope_his>1:
            df_copy_his["trend"]="Increasing"
        elif slope_his <-1:
            df_copy_his["trend"]="Decreasing"
        else:
            df_copy_his["trend"]="No Trend"
        time_points_for = [i for i in range(len(df_copy_for["quantity"]))]
        quantity_values_for = df_copy_for["quantity"]
        coefficients_for = np.polyfit(time_points_for, quantity_values_for, polynomial_degree)
        slope_for = coefficients_for[0]
        df_copy_for["slope_for"]=slope_for
        if slope_for>1:
            df_copy_for["trend"]="Increasing"
        elif slope_for <-1:
            df_copy_for["trend"]="Decreasing"
        else:
            df_copy_for["trend"]="No Trend"
        df_final = pd.concat([df_copy_his,df_copy_for])

        return df_final
@st.cache_data  
def recommend_products(user_id,df,top_n=5):
    if user_id not in df["user_id"].unique().tolist():
        # If the user is new, recommend the top-rated products
        top_rated_products = list(pd.Series(df.sort_values(by='rating', ascending=False)['product_id'].head(top_n)))
        return top_rated_products
    else:
        pivot_table = df.pivot_table(index='user_id', columns='product_id', values='rating', fill_value=0)
        user_ratings = pivot_table.loc[user_id]
        similarity = pivot_table.corrwith(user_ratings, axis=0)
        similar_products = similarity.sort_values(ascending=False).index[1:]  
        recommended_products = [product for product in similar_products if product not in df[df['user_id'] == user_id]['product_id']]
        return recommended_products[:top_n]
    
@st.cache_data
def generate_bedrock_response(prompt):
    # Initialize the Bedrock client
    bedrock = boto3.client(
        service_name='bedrock-runtime', 
        region_name="us-east-1"
    )

    # Define the parameters for the request
    body = json.dumps({
        "prompt": prompt,
        "max_tokens": 1000,
        "temperature": 0.75,
        "p": 0.01,
        "k": 0,
    })

    modelId = 'cohere.command-text-v14'
    accept = 'application/json'
    contentType = 'application/json'

    # Invoke the Bedrock model
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

    response_body = json.loads(response.get('body').read())

    return response_body['generations'][0]['text']


# Amazon Bedrock api call to stable diffusion
def generate_image(text, style):
    bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
        )
    """
    Purpose:
        Uses Bedrock API to generate an Image
    Args/Requests:
         text: Prompt
         style: style for image
    Return:
        image: base64 string of image
    """
    body = {
        "text_prompts": [{"text": text}],
        "cfg_scale": 10,
        "seed": 0,
        "steps": 50,
        "style_preset": style,
    }

    if style == "None":
        del body["style_preset"]

    body = json.dumps(body)

    modelId = "stability.stable-diffusion-xl"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_runtime.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("artifacts")[0].get("base64")
    return results

# Turn base64 string to image with PIL
def base64_to_pil(base64_string):
    """
    Purpose:
        Turn base64 string to image with PIL
    Args/Requests:
         base64_string: base64 string of image
    Return:
        image: PIL image
    """
    import base64

    imgdata = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(imgdata))
    return image



import json
import boto3

def generate_bedrock_jurasic(prompt):
    """
    Generate text using Amazon Bedrock.
    
    Args:
        prompt (str): The input text prompt.
        
    Returns:
        str: Generated text.
    """
    bedrock = boto3.client(
        service_name='bedrock-runtime', 
        region_name='us-east-1'
    )
    
    body = json.dumps({
        'prompt': prompt,
        'maxTokens': 1000,
        'temperature': 0.75,
        'topP': 1,
    })
    
    modelId = 'ai21.j2-ultra-v1'
    accept = 'application/json'
    content_type = 'application/json'
    accept = '*/*'
    
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=content_type)
    response_body = json.loads(response.get('body').read())
    
    return response_body["completions"][0]["data"]["text"]



