# LEGACY DOCUMENT
# The `Agent C` Compose stack


The Docker Compose file will stand up the services needed for development with the Agent C reference app and tools. There's a bit of configuration needed before launching it for the first time that we'll cover below.  

NOTE: This compose file is for DEVELOPMENT ONLY.  Do not run it on machines exposed directly to the internet.  OR at least change the postgres user and password in the compose file.

## What's in it:

### Zep
The [Zep](https://www.getzep.com/) stack makes up the majority of the compose file. It provides everything needed to manage chat sessions between users and an AI agent with some nice bells and whistles.  

- As the conversation between the user and agent progresses it can start consuming a lot of tokens, Zep helps manage this by using a cheap model to summarize the conversation on the fly. This allows us to supply the agent the last N full messages and a summary of the rest and save on context window space and token costs.
- Zep also does some limited extraction of things like Named Entities and attaches them as metadata on the messages.
- They expose `metadata` fields on the user and session objects for application use. Several agent tools make use of this metadata to store information.
- They have a full vector search API for searching chat sessions that can be exposed to the agent, if we provide the agent a way to get session IDs and summaries of them it can refer back to things that happened in other sessions.
- They have a vectorstore for file data but I don't recommend it. (Grab Weaviate instead)

NOTE:  Zep is exposed on port 8001 in this stack not 8000.


## Initial configuration
The file `example.env` contains placeholders for the values you will need to provide. In order to provide those values you will need to [download a small tool](https://github.com/getzep/zepcli/releases) from the Zep GitHub page. 

1. Download the appropriate version for your operating system and extract it to the docker folder of the project.
2. Copy the `example.env` file in the docker folder to `.env` in the same folder.  NOTE THE LEADING PERIOD.
3. In the terminal run `docker\zepcli.exe -i` on Windows, or `docker/zepcli -i` on Linux/OSX.
    1. Use `Secret` from the output to populate `ZEP_AUTH_SECRET` in the `.env` file.
    2. Use `JWT token` from the output to populate `ZEP_JWT_TOKEN` in the `.env` file.  This isn't strictly required but keeping the token next to the key ensures that if we change Zep servers in the app config we don't forget what this key was supposed to be..
4. Paste your Open AI API key to populate `ZEP_OPENAI_API_KEY` in the `.env` file.  Zep uses this to execute GPT-3.5-turbo (by default) in the background to produce summaries of the chat sessions on the fly.
5. The `zep.config.yml` file provides the ability to configure Zep beyond the defaults here if desired but you do not need to edit this file otherwise.


## Running the stack
NOTE: The services are only set to restart on failure, so you will need to launch them each reboot unless you modify the restart rules in the docker-compose.yml

On your first start up you should run `docker-compose up` so that you can verify that everything started correctly.  After that running `docker-compose up -d` will launch the stack in the background so it's not consuming a terminal.

NOTE: If you are on Windows and using docker via WSL, you'll need to launch compose from a WSL Bash prompt or the Zep config file will not get mapped and it will crash on start up.

**Rancher**: There's a setting you need to enable in Rancher so that you can connect to Zep via `localhost`.


