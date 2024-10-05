from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from chatbot import create_chatbot


def process_content(content):

    chunk_size = 2000
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    docs = [Document(page_content=chunk) for chunk in chunks]

    return docs


def get_tool(llm, content, template):

    prompt_template = template
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    meet_content = stuff_chain.run(content)

    return meet_content


if __name__ == "__main__":

    try:
        with open('meet_content.txt', 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        content = ""

    chatbot = create_chatbot()

    meeting_content = process_content(content)

    summary_template = """
    Tóm tắt toàn bộ nội dung của cuộc họp. Bao gồm các chủ đề chính, các quyết định quan trọng, và những vấn đề cần lưu ý. Trả lời theo mẫu sau:

    Chủ đề chính:
    - [Danh sách các chủ đề chính]
    Quyết định quan trọng:
    - [Danh sách các quyết định]
    Vấn đề cần lưu ý:
    - [Danh sách các vấn đề cần lưu ý]

    Nội dung cuộc họp: {text}
    """

    summary_result = get_tool(chatbot, meeting_content, summary_template)

    print("Kết quả tóm tắt cuộc họp:")
    print(summary_result)

    todo_template = """
    Dựa trên nội dung cuộc họp, tạo một danh sách công việc cần hoàn thành trong thời gian tới. Trả lời theo mẫu sau:

    Công việc:
    - [Tên công việc]
    Người phụ trách:
    - [Tên người phụ trách]
    Thời hạn hoàn thành:
    - [Ngày tháng cụ thể]

    Nội dung cuộc họp: {text}
    """

    todo_result = get_tool(chatbot, meeting_content, todo_template)

    print("Kết quả To-do list:")
    print(todo_result)

    planning_template = """
    Tạo một kế hoạch hành động cụ thể dựa trên nội dung cuộc họp, bao gồm các bước thực hiện, thời gian dự kiến cho từng giai đoạn và các nguồn lực cần thiết. Trả lời theo mẫu sau:

    Bước 1:
    - [Tên bước]
    Thời gian dự kiến:
    - [Thời gian cụ thể]
    Nguồn lực cần thiết:
    - [Danh sách nguồn lực]

    Bước 2:
    - [Tên bước]
    Thời gian dự kiến:
    - [Thời gian cụ thể]
    Nguồn lực cần thiết:
    - [Danh sách nguồn lực]

    Nội dung cuộc họp: {text}
    """

    planning_result = get_tool(chatbot, meeting_content, planning_template)

    print("Kết quả kế hoạch hành động:")
    print(planning_result)