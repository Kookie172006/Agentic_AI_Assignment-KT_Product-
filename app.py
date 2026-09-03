
import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain.tools import tool
from langchain.agents import create_agent
import faiss


# ============================================================
# 1. API KEY
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")


# ============================================================
# 2. GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="models/gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY
)


# ============================================================
# 3. KT GUIDE
# ============================================================

kt_guide_content = """
Welcome to InnovateCorp! This Knowledge Transfer (KT) guide is designed to help new employees navigate their initial weeks and understand key aspects of our operations. Our core values are Innovation, Collaboration, and Customer Focus.

**Team Structure:** You will be joining the 'Project Alpha' team, reporting to Sarah Chen, the Senior Project Manager. Your direct teammates include David Lee (Lead Developer), Maria Rodriguez (UI/UX Designer), and Tom Jackson (QA Engineer). Our team meetings are held every Monday at 10 AM in Conference Room 3, and daily stand-ups are at 9:30 AM via Google Meet.

**Key Tools & Software:** For project management, we use Jira for task tracking and Confluence for documentation. Our primary communication tool is Slack for instant messaging and Google Workspace for email and calendars. Development work is primarily done using Python and JavaScript, with code hosted on GitHub. Access to these tools will be granted within your first three days.

**Onboarding Process:** Your first week will focus on setup and introductions. You'll receive your laptop and login credentials on day one. HR will conduct an orientation session on Tuesday covering company policies, benefits, and payroll. You'll have one-on-one meetings with your team members throughout the week. By the end of your second week, you should have access to all necessary systems and have completed mandatory compliance training modules.

**Important Resources:** The company's internal knowledge base can be found at `internal.innovatecorp.com/kb`. This includes FAQs, best practices, and troubleshooting guides. For IT support, please submit a ticket via `support.innovatecorp.com` or call extension 5555. Health and wellness benefits information is available on the HR portal.

**Culture & Expectations:** InnovateCorp encourages a proactive and collaborative environment. We value open communication and continuous learning. Don't hesitate to ask questions; your team is here to support your growth. Performance reviews are conducted quarterly, and professional development courses are available through our 'InnovateLearn' platform.
"""


# ============================================================
# 4. CREATE DOCUMENTS
# ============================================================

kt_documents = [
    Document(page_content=kt_guide_content)
]


# ============================================================
# 5. SPLIT KT GUIDE
# ============================================================

kt_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

kt_chunks = kt_text_splitter.split_documents(kt_documents)


# ============================================================
# 6. CREATE EMBEDDINGS + FAISS
# ============================================================

embeddings_kt_guide = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

embedding_dim = len(
    embeddings_kt_guide.embed_query("hello world")
)

index_kt = faiss.IndexFlatL2(embedding_dim)


vector_store_kt_guide = FAISS(
    embedding_function=embeddings_kt_guide,
    index=index_kt,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

vector_store_kt_guide.add_documents(
    documents=kt_chunks
)


# ============================================================
# 7. RETRIEVAL TOOL
# ============================================================

@tool
def retrieve_kt_context(query: str):
    """
    Retrieve information from the InnovateCorp KT Guide
    to help answer employee questions.
    """

    retrieved_docs = vector_store_kt_guide.similarity_search(
        query,
        k=2
    )

    serialized = "\n\n".join(
        f"Content: {doc.page_content}"
        for doc in retrieved_docs
    )

    return serialized


# ============================================================
# 8. CREATE AGENT
# ============================================================

tools = [retrieve_kt_context]

prompt = """
You are an HR onboarding assistant for InnovateCorp.

You have access to a tool that retrieves information
from the official InnovateCorp KT Guide.

Always use the KT Guide when answering questions about
InnovateCorp.

Answer questions using only the information retrieved
from the KT Guide.

If the answer is not available in the KT Guide,
politely say that the information is not available
in the manual.

Do not invent or assume company information.
"""

kt_agent = create_agent(
    llm,
    tools,
    system_prompt=prompt
)


# ============================================================
# 9. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="InnovateCorp KT Assistant"
)


class Question(BaseModel):
    question: str


# ============================================================
# 10. HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <!DOCTYPE html>

    <html>

    <head>

        <title>InnovateCorp KT Assistant</title>

        <style>

            body {
                font-family: Arial, sans-serif;
                background: #f4f6f8;
                margin: 0;
                padding: 0;
            }

            .container {
                width: 700px;
                max-width: 90%;
                margin: 60px auto;
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            }

            h1 {
                text-align: center;
                color: #333;
            }

            .subtitle {
                text-align: center;
                color: #777;
                margin-bottom: 30px;
            }

            textarea {
                width: 100%;
                height: 100px;
                padding: 12px;
                font-size: 16px;
                border: 1px solid #ccc;
                border-radius: 8px;
                resize: none;
                box-sizing: border-box;
            }

            button {
                width: 100%;
                margin-top: 15px;
                padding: 12px;
                font-size: 16px;
                border: none;
                border-radius: 8px;
                background: #333;
                color: white;
                cursor: pointer;
            }

            button:hover {
                background: #555;
            }

            #answer {
                margin-top: 25px;
                padding: 20px;
                background: #f1f3f5;
                border-radius: 8px;
                white-space: pre-wrap;
                min-height: 50px;
            }

        </style>

    </head>


    <body>

        <div class="container">

            <h1>🤖 InnovateCorp KT Assistant</h1>

            <div class="subtitle">
                Ask questions about the InnovateCorp KT Guide
            </div>

            <textarea
                id="question"
                placeholder="Ask something like: Who should I report to on Project Alpha?"
            ></textarea>

            <button onclick="askQuestion()">
                Ask KT Assistant
            </button>

            <div id="answer">
                Your answer will appear here...
            </div>

        </div>


        <script>

            async function askQuestion() {

                const question =
                    document.getElementById("question").value;

                const answer =
                    document.getElementById("answer");

                if (!question.trim()) {

                    answer.innerText =
                        "Please enter a question.";

                    return;
                }

                answer.innerText =
                    "Thinking... 🤔";


                try {

                    const response = await fetch(
                        "/ask",
                        {
                            method: "POST",

                            headers: {
                                "Content-Type":
                                    "application/json"
                            },

                            body: JSON.stringify({
                                question: question
                            })
                        }
                    );


                    const data = await response.json();

                    answer.innerText = data.answer;

                }

                catch (error) {

                    answer.innerText =
                        "Something went wrong. Please try again.";

                }

            }

        </script>

    </body>

    </html>
    """


# ============================================================
# 11. ASK ENDPOINT
# ============================================================

@app.post("/ask")
def ask_question(data: Question):

    final_response = None

    for event in kt_agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": data.question
                }
            ]
        },
        stream_mode="values"
    ):

        message = event["messages"][-1]

        if hasattr(message, "content") and message.content:

            if isinstance(message.content, str):

                final_response = message.content

            elif isinstance(message.content, list):

                text_parts = []

                for item in message.content:

                    if (
                        isinstance(item, dict)
                        and item.get("type") == "text"
                    ):

                        text_parts.append(
                            item.get("text", "")
                        )

                if text_parts:

                    final_response = " ".join(text_parts)


    return {
        "answer": final_response or
        "Sorry, I couldn't generate an answer."
    }
