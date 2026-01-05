import google.generativeai as genai
import os
import time

# 1. Setup
API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=API_KEY)

print("Listing available models...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
# 2. Upload the VMC Protocol PDF
# The API lets you upload files directly (up to 2GB) for the model to read.
print("Uploading file...")
vmc_pdf = genai.upload_file(path="VMC-Upper computer_V3.0_0411.pdf")

# Wait for processing (usually instant for small PDFs)
while vmc_pdf.state.name == "PROCESSING":
    print("Processing file...")
    time.sleep(2)
    vmc_pdf = genai.get_file(vmc_pdf.name)

if vmc_pdf.state.name == "FAILED":
    raise ValueError(f"File processing failed: {vmc_pdf.state.name}")

print(f"File ready: {vmc_pdf.uri}")

# 3. Initialize the Model (Gemini 1.5 Flash)
# Flash is fast, free-tier eligible, and has 1M token context (plenty for manuals)
model = genai.GenerativeModel(
    model_name="gemini-3-flash-preview",
    system_instruction="You are an expert on this VMC Protocol. Answer queries based strictly on the provided file. if the information is not found in the file, return not found."
)

# 4. Start Chatting (NotebookLM Experience)
chat = model.start_chat(
    history=[
        {
            "role": "user",
            "parts": [vmc_pdf, "Analyze this protocol document."]
        }
    ]
)

# 5. Test it
response = chat.send_message("who is the ceo of xy company?")
print(f"Answer: {response.text}")

# You can now wrap this 'chat.send_message' in your Flask API!