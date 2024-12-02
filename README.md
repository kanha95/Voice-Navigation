# Few-Shot Semantic Prompt with LangChain and Gradio

This repository implements a semantic similarity-based example selector and few-shot prompting for generating AI responses using OpenAI's GPT models. The application is built using LangChain, Chroma for vector storage, and Gradio for a user-friendly interface. The model provides tailored responses by leveraging similar examples stored in a vector database.

---

## **Features**

1. **Semantic Similarity Example Selector:**
   - Utilizes a Chroma vector store and OpenAI embeddings to find the most relevant examples based on the input query.

2. **Few-Shot Prompting:**
   - Dynamically generates responses using examples that follow a specific pattern or style.

3. **Dynamic Date Handling:**
   - Automatically adjusts responses with accurate date ranges (e.g., last 7 days, last quarter).

4. **Gradio Interface:**
   - A web-based interface for interacting with the model, allowing users to input queries and receive AI-generated responses.

---

## **Repository Structure**

- **`app.py`:** The main script containing the implementation of the LangChain pipeline and Gradio interface.
- **`chroma_db/`:** Directory to store Chroma vector database (created at runtime).
- **`.env`:** Environment variables file for storing the OpenAI API key securely.

---

## **Setup Instructions**

### 1. **Clone the Repository**
```bash
git clone <repository_url>
cd <repository_folder>
```

### 2. **Install Dependencies**
Ensure you have Python installed, then install the required dependencies:
```bash
pip install -r requirements.txt
```

### 3. **Set OpenAI API Key**
- Create a `.env` file in the project directory:
  ```plaintext
  OPENAI_KEY=your_openai_api_key
  ```
- The script will automatically load this API key.

### 4. **Run the Application**
Start the Gradio interface:
```bash
python app.py
```
The application will be accessible at `http://127.0.0.1:7860` by default.

---

## **Usage**

### **Input**
- Enter a query in natural language. For example:
  - "What steps are involved in refreshing all records from the electronic health records?"
  - "Showcase the sales data for the last quarter."

### **Output**
- The model generates a response tailored to your query, incorporating examples with semantic relevance and dynamically calculated date ranges where applicable.

---

## **Customization**

### **1. Chroma Vector Store**
- Add or update examples in the `chroma_db/` directory to refine the model’s responses.
- Use `OpenAIEmbeddings` for new data embeddings:
  ```python
  embeddings = OpenAIEmbeddings(api_key=OPENAI_KEY)
  vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
  ```

### **2. Few-Shot Prompt Template**
- Update the `few_shot_prompt` or the `final_prompt` in `app.py` to change how examples are used and formatted.

### **3. Model Parameters**
- Adjust temperature, model type, or other parameters in the `ChatOpenAI` configuration:
  ```python
  chain = final_prompt | ChatOpenAI(openai_api_key=OPENAI_KEY, model='gpt-4', temperature=0.7)
  ```

---

## **Dependencies**

- Python 3.7+
- LangChain
- Chroma
- OpenAI Python Library
- Gradio
- dotenv

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## **Notes**

1. **Persistent Database:**
   - The Chroma vector store is stored locally in the `./chroma_db` directory. Ensure this directory is available for the application to function.

2. **Production Deployment:**
   - For production use, consider securing the API key, optimizing database queries, and deploying on a robust server.

3. **Dynamic Date Handling:**
   - The system automatically replaces date placeholders with accurate ranges for relative queries (e.g., "last week").

---

## **License**
This project is licensed under the MIT License.

---
