from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8081/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

video_url = "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4"
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
## Use video url in the payload
chat_completion_from_url = client.chat.completions.create(
    messages=[{
        "role":
        "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this video?" # Nano model assume <image> placeholder is in the prompt
                #  "text": "<video>What's in this video and whats on the tv?" # Nano model assume <video> placeholder is in the prompt
            },
            {
                # "type": "video_url",
                # "video_url": {
                #     "url": video_url
                # },

                "type": "image_url",
                "image_url": {
                    "url": image_url
                },
            },
        ],
    }],
    # model="/home/dafrimi/projects/models/working_13p41",
    model="/home/dafrimi/projects/models/vlm_update_ckpt",
    max_completion_tokens=64,
)

result = chat_completion_from_url.choices[0].message.content
print("Chat completion output from image url:", result)