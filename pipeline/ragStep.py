from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

from secret.config import OPENAI_KEY


class OpenAIError(Exception):
    pass


def rag_step(size, overlap, context, ragInstruction, quantity):
    """Makes the RAG step for the given process
    
    Args:
        size (int): Size of the chunks for the character text splitter
        overlap (int): Size of the overlap of the chunks
        context (str): Text to be splitted and then retrieve the most important information
        ragInstruction (str): The instruction given to the RAG process to search for in the context
        quantity (int): The amount of most similar chunks to consider
    
    Returns:
        A summary made from documents more similar to the described instruction (str)
    """
    # Create an OpenAI model instance with LangChain (model and embeddings)
    try:
        llm = OpenAI(api_key=OPENAI_KEY)
        embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY)
    except Exception as e:
        raise OpenAIError(f"OpenAI error: {e}")

    # Processing the files
    processed_contents = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=size,
                                              chunk_overlap=overlap)

    # Getting the text directly
    split_content = splitter.split_text(context)
    for chunk in split_content:
        # Append the split content to the list
        processed_contents.append(Document(page_content=chunk, metadata={"source": "manual_description"}))

    # Indexing the fragments using FAISS
    index = FAISS.from_documents(processed_contents, embeddings)

    # Define the QA chain
    try:
        qa_chain = load_qa_chain(llm, chain_type="stuff")
    except Exception as e:
        raise OpenAIError(f"OpenAI error at QA chain: {e}")

    # Retrieve the most relevant documents using the query/RAG instruction
    retrieved_docs = index.similarity_search(ragInstruction, k=quantity)

    # Prepare the inputs for the chain
    inputs = {
        "input_documents": retrieved_docs,
        "question": ragInstruction
    }

    # Final summary
    final_summary = qa_chain.invoke(inputs)

    return final_summary["output_text"]
