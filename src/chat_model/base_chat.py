import os
import re
import warnings
from langchain import hub
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_astradb import AstraDBVectorStore
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

warnings.filterwarnings("ignore")

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ASTRA_DB_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

searcher = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="dateherk",
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
)

llm = ChatGroq(model="mixtral-8x7b-32768")

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

retriever = searcher.as_retriever()

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = (
    "You are a conversational assistant. Use the retrieved context and "
    "the conversation history to answer questions and personalize responses. "
    "If the user shares their name, remember it and use it appropriately."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
react_docstore_prompt = hub.pull("hwchase17/react")


def search_wikipedia(query):
    from wikipedia import summary

    try:
        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that."


tools = [
    Tool(
        name="Answer Question through RAG",
        func=lambda input, **kwargs: rag_chain.invoke(
            {
                "input": input,
                "chat_history": kwargs.get("chat_history", []),
            }
        ),
        description="useful for when you need to answer questions about the context",
    ),
    Tool(
        name="wikipedia",
        func=search_wikipedia,
        description="useful when you cannot find answers about the context",
    ),
]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_docstore_prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,
)


def gen(query, chat_history):
    response = agent_executor.invoke({"input": query, "chat_history": chat_history})
    return response["output"]


prompt_template = PromptTemplate(
    input_variables=["text", "attribute_type"],
    template="Extract a list of {attribute_type} mentioned in the following text, ensuring only positive or neutral attributes are included: {text}. Provide only the words in a Python list.",
)


def char(input_text, attribute_type):
    formatted_prompt = prompt_template.format(
        text=input_text, attribute_type=attribute_type
    )
    response = llm.chat.completions.create(
        messages=[{"role": "user", "content": formatted_prompt}],
        model="llama3-8b-8192",
    )
    output = response.choices[0].message.content.strip()
    match = re.search(r"\[.*\]", output)
    return eval(match.group(0)) if match else ["efforts"]
