# WebChat-Assistant

## Project Overview

Chat with Websites is an interactive Streamlit application that allows users to have a conversational AI experience with the content of any web page. By leveraging advanced language models and retrieval techniques, this app enables you to ask questions and get context-aware responses directly from a website's content.

## 🚀 Features

- **Web Content Extraction**: Scrape and process content from any public website
- **Conversational AI**: Engage in a dynamic, context-aware conversation about the website's content
- **Retrieval-Augmented Generation (RAG)**: Provide accurate, context-based responses
- **Chat History Tracking**: Maintain conversation context across multiple interactions

## 🛠 Technologies Used

- Streamlit
- LangChain
- OpenAI GPT
- ChromaDB (Vector Store)
- Python

## 📋 Prerequisites

- Python 3.8+
- OpenAI API Key
- Libraries listed in `requirements.txt`

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/deepmehta27/WebChat-Assistant.git
cd src
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
- Create a `.env` file in the project root
- Add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## 🚦 How to Run

```bash
streamlit run app.py
```

## 🔍 How to Use

1. Open the Streamlit app
2. Enter a website URL in the sidebar
3. Start chatting about the website's content
4. Ask questions, get context-aware responses

## 💡 Usage Tips

- Use complete, specific questions
- Ensure the website URL is publicly accessible
- Some complex websites might have limited content extraction
- Websites blocking web scraping may not work effectively

## 🚧 Limitations

- Depends on OpenAI's API and embedding capabilities
- Web scraping might not work for all websites
- Response quality depends on website content complexity

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This extension is licensed under the MIT License. Please see the [src/LICENSE.txt](./src/LICENSE.txt) file for details.

This project may also include third-party libraries or resources that are subject to their own licenses. Please refer to the third-party notices file for additional copyright notices and license terms applicable to portions of the software.

## 🙌 Acknowledgments

- Streamlit
- LangChain
- OpenAI
- ChromaDB
