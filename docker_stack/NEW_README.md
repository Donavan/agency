# ONLY USE IF YOU KNOW WHAT YOU ARE DOING
## Weaviate Setup
Set up your environment variables in several places
- `main app\.env` 
```dotenv
WEAVIATE_API_KEY='Your API Key'
```
- `docker_stack\.env`
```dotenv
AUTHENTICATION_APIKEY_ALLOWED_KEYS='use the same key as WEAVIATE_API_KEY in main app .env'
AUTHENTICATION_APIKEY_USERS='provide a user name like weaviate-dev-user or admin'
```
The Weaviate tool instantiates the client with code like
```python
self.client: weaviate.WeaviateClient = kwargs.get(
            'client',
            weaviate.connect_to_local(headers={"X-Openai-Api-Key": os.getenv("OPENAI_API_KEY"),
```
