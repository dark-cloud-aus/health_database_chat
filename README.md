This is part of my project for the Medhack 2025 project at Monash University 2025.  

Before you begin you will need to setup a Pinecone free account. Obtain the APi key and OpenAI API as well.  Insert API keys into your .env file

This a python tool that is used to analyze hospital patient data. The data I used was in csv format. Add data files the 'data' folder.  Setup a virtual enviroment, install the required packages and run main.py

The program pushes the data files to pinecone by http, pinecone then vectorises that info and then a chat bot is launched which allows you to 'talk' to the data.
