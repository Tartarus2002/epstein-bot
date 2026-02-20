import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def extract_text_from_image(image_url: str) -> str:
    """
    Uses NVIDIA NIM API (Llama 3.2 Vision) to extract/transcribe
    all visible text from a document image.

    Args:
        image_url: Public URL of the image to process.

    Returns:
        Extracted text string, or empty string on failure.
    """
    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY"),
        )

        response = client.chat.completions.create(
            model="nvidia/nemotron-nano-12b-v2-vl",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe all visible text in this document image perfectly. If there is no text, just say 'No text'.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            max_tokens=1024,
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[Vision Error] Failed to extract text from {image_url!r}: {e}")
        return ""
