# ============================================================
# UNCLE SOJI'S SUYA SPOT — AI Customer Assistant
# ============================================================

# ── PHASE 1: SETUP ──────────────────────────────────────────

import os
import json
import time
import random
import string
import smtplib
import uuid
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

import gradio as gr
import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI as OpenAIClient
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

# Load API keys from environment (set as secrets in Hugging Face Spaces)
openai_client = OpenAIClient(api_key=os.environ.get("OPENAI_API_KEY"))
pc            = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

Settings.llm         = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size  = 512
Settings.chunk_overlap = 50

print("✅ Setup complete")

# ── PHASE 2: PINECONE ────────────────────────────────────────

index_name = "uncle-soji-suya"

if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

pinecone_index = pc.Index(index_name)

if pinecone_index.describe_index_stats()['total_vector_count'] == 0:
    vector_store    = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    documents       = SimpleDirectoryReader("suya_docs").load_data()
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    print(f"✅ Indexed {len(documents)} documents into Pinecone")
else:
    print("✅ Loaded existing Pinecone index")

# ── PHASE 3: DOCUMENT SEARCH ─────────────────────────────────

def query_documents(query: str, topic_hint: str = "") -> str:
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=f"{topic_hint} {query}".strip()
    ).data[0].embedding

    results = pinecone_index.query(vector=embedding, top_k=3, include_metadata=True)
    texts   = [m.metadata.get("text", "") for m in results.matches if m.metadata]
    return "\n\n".join(texts) if texts else "No information found."

print("✅ Document search ready")

# ── PHASE 4: GOOGLE SHEETS ───────────────────────────────────

# Service account JSON is stored as a secret string in Hugging Face Spaces
_sa_json  = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
_sa_dict  = json.loads(_sa_json)
_scopes   = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
_creds    = Credentials.from_service_account_info(_sa_dict, scopes=_scopes)
gc        = gspread.authorize(_creds)

try:
    spreadsheet = gc.open("Uncle Soji Orders")
except:
    spreadsheet = gc.create("Uncle Soji Orders")

try:
    worksheet = spreadsheet.worksheet("Orders")
except:
    worksheet = spreadsheet.add_worksheet(title="Orders", rows=1000, cols=10)
    worksheet.append_row(["Timestamp","Order ID","Customer Email","Items",
                          "Subtotal","Delivery Location","Delivery Fee","Total","Status","Notes"])

print("✅ Google Sheets ready")

# ── PHASE 5: ORDER FUNCTIONS ─────────────────────────────────

def place_order(items_json: str, delivery_location: str, customer_email: str) -> str:
    menu_prices = {
        "beef suya": 2500,   "chicken suya": 2000, "gizzard suya": 1800,
        "ram suya":  3500,   "turkey suya":  3000, "mixed grill":  4000,
        "family pack": 12000, "party pack":  35000
    }
    delivery_fees = {
        "yaba": 1000, "surulere": 1500, "ikeja": 1500, "mushin": 1500,
        "victoria island": 2000, "vi": 2000, "ikoyi": 2000, "lekki": 2000, "ajah": 2000
    }

    try:
        items = json.loads(items_json)
    except:
        return "Could not parse order items. Please try again."

    subtotal, item_details = 0, []
    for item in items:
        name  = item['item'].lower()
        qty   = item['quantity']
        price = next((p for k, p in menu_prices.items() if k in name or name in k), None)
        if not price:
            return f"Item not found: {item['item']}"
        item_details.append({"item": name, "quantity": qty, "unit_price": price, "total": price * qty})
        subtotal += price * qty

    is_pickup    = delivery_location.lower() == "pickup"
    delivery_fee = 0

    if not is_pickup:
        delivery_fee = next((f for zone, f in delivery_fees.items() if zone in delivery_location.lower()), 0)
        if not delivery_fee:
            return f"We don't deliver to {delivery_location}. Available: Yaba, Surulere, Ikeja, Mushin, Victoria Island, Ikoyi, Lekki, Ajah"

    total    = subtotal + delivery_fee
    order_id = f"SO-{datetime.now().strftime('%Y%m%d')}-{''.join(random.choices(string.digits, k=4))}"
    items_str = ", ".join([f"{i['quantity']}x {i['item']}" for i in item_details])

    worksheet.append_row([
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), order_id, customer_email,
        items_str, subtotal, "Pickup" if is_pickup else delivery_location,
        delivery_fee, total, "Pending", ""
    ])

    _send_emails(order_id, item_details, subtotal, delivery_location, delivery_fee, total, customer_email)

    lines        = "\n".join([f"  {i['quantity']}x {i['item']} — NGN {i['total']:,}" for i in item_details])
    delivery_line = f"Delivery to {delivery_location}: NGN {delivery_fee:,}" if not is_pickup else "Pickup (no delivery fee)"

    return (f"Order confirmed! ✅\n\nOrder ID: {order_id}\n\n{lines}\n\n"
            f"Subtotal: NGN {subtotal:,}\n{delivery_line}\nTotal: NGN {total:,}\n\n"
            f"Confirmation sent to {customer_email}. Ready in 30–45 minutes.")

def _send_emails(order_id, item_details, subtotal, delivery_location, delivery_fee, total, customer_email):
    gmail_address  = os.environ.get("GMAIL_ADDRESS")
    gmail_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not gmail_address or not gmail_password:
        return

    lines         = "\n".join([f"  {i['quantity']}x {i['item']} — NGN {i['total']:,}" for i in item_details])
    delivery_line = f"Delivery to {delivery_location}: NGN {delivery_fee:,}" if delivery_location.lower() != "pickup" else "Pickup"

    for to_email, subject, body in [
        (customer_email,
         f"Order Confirmation #{order_id} — Uncle Soji's Suya Spot",
         f"Order ID: {order_id}\n\n{lines}\n\nSubtotal: NGN {subtotal:,}\n{delivery_line}\n"
         f"Total: NGN {total:,}\n\nReady in 30–45 minutes.\n\nUncle Soji's Suya Spot | 0803-555-7892"),
        (gmail_address,
         f"NEW ORDER #{order_id} — NGN {total:,}",
         f"Order ID: {order_id}\nCustomer: {customer_email}\n\n{lines}\n\nTotal: NGN {total:,} | Status: PENDING")
    ]:
        try:
            msg             = MIMEMultipart()
            msg['From']     = gmail_address
            msg['To']       = to_email
            msg['Subject']  = subject
            msg.attach(MIMEText(body, 'plain'))
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(gmail_address, gmail_password)
            server.send_message(msg)
            server.quit()
        except:
            pass

print("✅ Order functions ready")

# ── PHASE 6: AGENT ───────────────────────────────────────────

conversation_history = {}

TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "search_menu",     "description": "Search menu items, prices, combos, sides and drinks",                 "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "search_hours",    "description": "Search operating hours, delivery hours and kitchen closing times",    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "search_location", "description": "Search restaurant address, landmarks, parking and directions",        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "search_policy",   "description": "Search delivery areas, fees, minimum orders and payment methods",    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "search_offers",   "description": "Search promotions, discounts and loyalty program",                   "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "search_about",    "description": "Search restaurant story, hygiene standards and contact information", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "place_order",     "description": "Place a confirmed customer order. Only call when customer explicitly confirms.",
        "parameters": {"type": "object", "properties": {
            "items_json":        {"type": "string"},
            "delivery_location": {"type": "string"},
            "customer_email":    {"type": "string"}
        }, "required": ["items_json", "delivery_location", "customer_email"]}}}
]

TOOL_MAP = {
    "search_menu":     lambda q: query_documents(q, "menu items prices food"),
    "search_hours":    lambda q: query_documents(q, "operating hours opening closing times"),
    "search_location": lambda q: query_documents(q, "restaurant address location directions"),
    "search_policy":   lambda q: query_documents(q, "delivery policy fees minimum order payment"),
    "search_offers":   lambda q: query_documents(q, "promotions discounts special offers deals"),
    "search_about":    lambda q: query_documents(q, "about restaurant story contact hygiene"),
    "place_order":     lambda **kw: place_order(**kw)
}

SYSTEM_PROMPT = """You are the AI assistant for Uncle Soji's Suya Spot, a suya restaurant in Yaba, Lagos.

BEHAVIOUR:
- Warm, friendly and concise
- Always use search tools — never assume prices or hours from memory
- Only call place_order after customer confirms their full order

ORDERING FLOW:
1. Clarify items and quantities
2. Ask delivery or pickup
3. If delivery, ask for area
4. Confirm order summary with total
5. Ask for email
6. Call place_order"""

def run_agent(user_id: str, message: str) -> str:
    history  = conversation_history.get(user_id, [])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history[-6:]:
        messages.append({"role": "user",      "content": turn["query"]})
        messages.append({"role": "assistant", "content": turn["response"]})
    messages.append({"role": "user", "content": message})

    while True:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto"
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]
        })

        for tc in msg.tool_calls:
            args   = json.loads(tc.function.arguments)
            result = TOOL_MAP[tc.function.name](**args) if tc.function.name == "place_order" else TOOL_MAP[tc.function.name](args["query"])
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

def chat(user_id: str, message: str) -> str:
    response = run_agent(user_id, message)
    conversation_history.setdefault(user_id, []).append({"query": message, "response": response})
    return response

print("✅ Agent ready")

# ── PHASE 7: VOICE ───────────────────────────────────────────

def transcribe_audio(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        return openai_client.audio.transcriptions.create(
            model="whisper-1", file=f, language="en"
        ).text

def speak_response(text: str, output_path: str = "/tmp/response.mp3") -> str:
    openai_client.audio.speech.create(
        model="tts-1", voice="nova", input=text
    ).stream_to_file(output_path)
    return output_path

print("✅ Voice ready")

# ── PHASE 8: GRADIO INTERFACE ────────────────────────────────

def get_session_id():
    return str(uuid.uuid4())

def handle_text(message, email, session_id, history):
    if not message.strip():
        return history, ""
    user_id  = email.strip() if email.strip() else session_id
    response = chat(user_id, message)
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": response})
    return history, ""

def handle_voice(audio_path, email, session_id, history):
    if not audio_path:
        return history, None
    user_id    = email.strip() if email.strip() else session_id
    transcript = transcribe_audio(audio_path)
    response   = chat(user_id, transcript)
    audio_out  = speak_response(response, f"/tmp/response_{session_id[:8]}.mp3")
    history.append({"role": "user",      "content": f"🎤 {transcript}"})
    history.append({"role": "assistant", "content": response})
    return history, audio_out

with gr.Blocks(title="Uncle Soji's Suya Spot") as demo:

    session_id = gr.State(get_session_id)

    gr.Markdown("## 🔥 Uncle Soji's Suya Spot")

    with gr.Row(equal_height=True):

        with gr.Column(scale=2, min_width=400):
            chatbot     = gr.Chatbot(type="messages", height=380, show_label=False, allow_tags=False)
            email_input = gr.Textbox(placeholder="Your email (needed to place an order)", show_label=False)
            with gr.Row():
                text_input = gr.Textbox(placeholder="Ask me anything...", show_label=False, scale=5)
                send_btn   = gr.Button("Send", variant="primary", scale=1)

        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### 🎤 Voice")
            gr.Markdown("Record your message and get a spoken response back.")
            voice_input  = gr.Audio(sources=["microphone"], type="filepath", label="Your message")
            voice_output = gr.Audio(label="Response", autoplay=True)

    send_btn.click(handle_text, [text_input, email_input, session_id, chatbot], [chatbot, text_input])
    text_input.submit(handle_text, [text_input, email_input, session_id, chatbot], [chatbot, text_input])
    voice_input.stop_recording(handle_voice, [voice_input, email_input, session_id, chatbot], [chatbot, voice_output])

demo.launch()
