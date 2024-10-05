from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from chatbot import create_chatbot


def process_content(content):

    chunk_size = 2000
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    docs = [Document(page_content=chunk) for chunk in chunks]

    return docs


def get_summary(content):
    
    chatbot = create_chatbot()
    meeting_content = process_content(content)

    prompt_template = """
        Viết tóm tắt thật chi tiết và đảm bảo đủ lượng thông tin cho cuộc họp
        Dưới đây là nội dung cuộc họp:
        "{text}"
        Nội dung cuộc họp được tóm tắt phải được trình bày theo mẫu sau:
        # Tóm Tắt Cuộc Họp
        ## Chủ đề cuộc họp
        ## Mục đích
        ## Nội dung
        ## Các điểm chính
    """

    prompt = PromptTemplate.from_template(prompt_template)


    llm_chain = LLMChain(llm=chatbot, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    summary_meet_content = stuff_chain.run(meeting_content)

    return summary_meet_content


if __name__ == "__main__":

    try:
        with open('meet_content.txt', 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        content = ""

    summary = get_summary(content)
    print(summary)