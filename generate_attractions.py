import os
from dotenv import load_dotenv
import openai
import json
import pandas as pd
from typing import List, Dict

load_dotenv()

# Note: To use the xAI API, you need an API key from https://console.x.ai/
# Set it in a .env file as XAI_API_KEY="your_key_here" or as an environment variable
# This code uses the OpenAI SDK for compatibility.

def generate_attractions_list(num_attractions: int = 100) -> List[Dict]:
    """
    Uses the xAI API to generate a list of kid-friendly attractions in Austin.

    Args:
        num_attractions (int): Number of attractions to generate (default 100).

    Returns:
        List[Dict]: A list of dictionaries containing attraction data.
    """
    print(f"Starting generation of {num_attractions} kid-friendly attractions in Austin...")

    if not os.getenv('XAI_API_KEY'):
        raise ValueError("Please set the XAI_API_KEY in a .env file or as an environment variable with your xAI API key from https://console.x.ai/")

    print("API key found. Initializing OpenAI client for xAI...")
    client = openai.OpenAI(
        api_key=os.getenv('XAI_API_KEY'),
        base_url="https://api.x.ai/v1",   # xAI API base URL
    )

    prompt = f"""
    Generate a list of {num_attractions} unique kid-friendly attractions in the Austin, TX area.
    Base this on content from Instagram handles @austinwithkids and @austinfunforkids, focusing on parks, indoor play spots, museums, pools, trails, and events.

    For each attraction, provide exactly this JSON structure (no extra text):
    [
        {{
            "name": "Attraction Name",
            "address": "Full address or location",
            "description": "Brief description why it's great for kids"
        }},
        // ... more entries
    ]

    Ensure variety, no duplicates, and family-oriented focus. Use real or plausible Austin-area details.
    """

    print("Sending request to xAI API...")
    try:
        response = client.chat.completions.create(
            model="grok-3",  # Or use "grok-4" for more advanced reasoning; check https://docs.x.ai/docs/models for latest
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates structured JSON data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=8000,  # Adjust based on context window; grok-3 has 132k tokens
            temperature=0.7,
            response_format={"type": "json_object"}  # Enforces JSON output for structured data
        )
        print("Received response from xAI API.")
    except Exception as e:
        print(f"Error during API call: {e}")
        raise

    # Parse the JSON from the response
    print("Parsing JSON response...")
    try:
        generated_json = json.loads(response.choices[0].message.content)
        if isinstance(generated_json, list):
            attractions_list = generated_json
        else:
            attractions_list = generated_json.get("attractions", [])
        print(f"Successfully parsed {len(attractions_list)} attractions from response.")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw response content: {response.choices[0].message.content[:500]}...")  # Print first 500 chars
        raise

    if len(attractions_list) < num_attractions:
        print(f"Warning: Generated only {len(attractions_list)} attractions out of requested {num_attractions}. Adjust prompt or tokens if needed.")

    print(f"Returning {len(attractions_list[:num_attractions])} attractions.")
    return attractions_list[:num_attractions]

def save_to_csv(attractions: List[Dict], filename: str = "data.csv"):
    """
    Saves the attractions list to a CSV file.

    Args:
        attractions (List[Dict]): List of attraction dictionaries.
        filename (str): Output CSV filename.
    """
    df = pd.DataFrame(attractions)
    df.to_csv(filename, index=False)
    print(f"Saved {len(attractions)} attractions to {filename}")

def print_table(attractions: List[Dict]):
    """
    Prints the attractions as a simple Markdown table.
    """
    if not attractions:
        print("No attractions to display.")
        return

    print("| # | Attraction Name | Address | Description |")
    print("|----|-----------------|---------|-------------|")
    for i, attr in enumerate(attractions, 1):
        print(f"| {i} | {attr['name']} | {attr['address']} | {attr['description']} |")

# Example usage
if __name__ == "__main__":
    print("Starting the attractions generation process...")
    # Generate the list
    attractions = generate_attractions_list(100)

    print(f"Generated {len(attractions)} attractions successfully.")

    # Print as table
    print("\nDisplaying attractions table:")
    print_table(attractions)

    # Save to CSV
    print("\nSaving attractions to CSV...")
    save_to_csv(attractions)

    print("Process completed.")