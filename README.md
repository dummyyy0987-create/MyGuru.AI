# Confluence AI Chatbot

A beautiful and intelligent chatbot that searches through your Confluence documentation (including all spaces and PDFs) to answer questions. Built with Streamlit, Azure OpenAI GPT-4o, and FAISS vector search.

## Features

✨ **Comprehensive Search**
- Searches across all Confluence spaces
- Includes content from PDF attachments
- Extracts and displays related GitHub repository links

🤖 **AI-Powered Responses**
- Uses Azure OpenAI GPT-4o for intelligent answers
- Provides context-aware responses
- Shows source references for transparency

🎨 **Beautiful UI**
- Clean and modern Streamlit interface
- Real-time chat experience
- Displays sources and relevance scores

🔍 **Semantic Search**
- FAISS vector store for fast similarity search
- Azure text-embedding-3-small for embeddings
- Retrieves most relevant content chunks

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI account with:
  - GPT-4o deployment
  - text-embedding-3-small deployment
- Confluence Cloud instance with API access

## Installation

1. **Clone the repository or navigate to the project directory**

```bash
cd AIGuru
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Copy the `.env.example` file to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small

# Confluence Configuration
CONFLUENCE_URL=https://your-domain.atlassian.net
CONFLUENCE_USERNAME=your_email@example.com
CONFLUENCE_API_TOKEN=your_confluence_api_token
```

### Getting Confluence API Token

1. Go to https://id.atlassian.com/manage-profile/security/api-tokens
2. Click "Create API token"
3. Give it a label and copy the token
4. Use your email as username and the token as password

### Setting up Azure OpenAI

1. Create an Azure OpenAI resource in the Azure portal
2. Deploy GPT-4o model
3. Deploy text-embedding-3-small model
4. Copy the endpoint and API key

## Setup

Before running the chatbot, you need to index your Confluence documentation:

```bash
python setup_index.py
```

This script will:
1. Connect to your Confluence instance
2. Fetch all pages from all spaces
3. Download and process PDF attachments
4. Create embeddings using Azure OpenAI
5. Build a FAISS vector index

**Note:** This process may take several minutes depending on the size of your Confluence instance.

## Running the Chatbot

Once the index is created, start the Streamlit app:

```bash
streamlit run app.py
```

The chatbot will open in your default browser at `http://localhost:8501`

## Usage

1. **Ask Questions**: Type your question in the text input field
2. **View Responses**: The AI will provide answers based on your Confluence documentation
3. **Check Sources**: See which Confluence pages were used to generate the answer
4. **Find Related Repos**: GitHub repository links mentioned in the documentation will be displayed
5. **Clear History**: Use the sidebar button to start a new conversation

## Project Structure

```
AIGuru/
├── app.py                  # Main Streamlit application
├── config.py              # Configuration management
├── confluence_fetcher.py  # Confluence API integration
├── vector_store.py        # FAISS vector store implementation
├── setup_index.py         # Index creation script
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## How It Works

1. **Data Fetching**: `confluence_fetcher.py` connects to Confluence and retrieves all pages and PDFs
2. **Text Processing**: Content is split into chunks for better semantic search
3. **Embeddings**: Azure OpenAI creates vector embeddings for each chunk
4. **Indexing**: FAISS stores the vectors for fast similarity search
5. **Query Processing**: User queries are embedded and matched against the index
6. **Response Generation**: GPT-4o generates answers using the most relevant chunks as context

## Troubleshooting

**Import errors**: Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

**Index not found**: Run the setup script:
```bash
python setup_index.py
```

**Confluence connection issues**: 
- Verify your credentials in `.env`
- Ensure your API token is valid
- Check your Confluence URL format

**Azure OpenAI errors**:
- Verify your API key and endpoint
- Ensure your deployments are named correctly
- Check you have sufficient quota

## Updating the Index

To refresh your index with new Confluence content:

```bash
python setup_index.py
```

This will fetch the latest documents and rebuild the index.

## Performance Tips

- The initial indexing can take time for large Confluence instances
- Adjust chunk sizes in `vector_store.py` if needed
- Increase `k` parameter in search for more context (but slower responses)
- Use FAISS GPU version for faster search on large datasets

## Contributing

Feel free to submit issues or pull requests to improve the chatbot!

## License

MIT License
