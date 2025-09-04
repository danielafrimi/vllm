

from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4"
image_url = "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"
## Use video url in the payload
chat_completion_from_url = client.chat.completions.create(
    messages=[{
        "role":
        "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this video?"
            },
            # {
            #     "type": "video_url",
            #     "video_url": {
            #         "url": video_url
            #     },

            # },
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        ],
    }],
    # model="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    model="OpenGVLab/InternVL2-2B",
    # model="/home/dafrimi/projects/models/working_13p41",
    max_completion_tokens=64,
)

result = chat_completion_from_url.choices[0].message.content
print("Chat completion output from image url:", result)