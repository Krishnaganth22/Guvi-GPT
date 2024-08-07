## Text Generation App with GPT-2
This repository contains a Streamlit web application for generating text using a fine-tuned GPT-2 model. The application allows users to input prompts and generate text based on these prompts, using a GPT-2 model that has been fine-tuned with custom data. The app also includes user authentication and logging functionality.

# Table of Contents
Installation
Usage
File Structure
Code Explanation
Contributing
License
#Installation
To run this application, you need to have Python installed. Follow the steps below to set up the environment and install the necessary packages.

1.Clone the repository
2.Create a virtual environment and activate it
3.Install the required packages:
4.Set up Hugging Face authentication:

Usage
Preprocess your data and fine-tune the GPT-2 model by running the following script
Run the Streamlit app
Open your web browser and navigate to the URL provided by Streamlit to interact with the app.

# Code Explanation
Preprocessing and Fine-Tuning (train_model.py)
Preprocess Data: The preprocess_data function tokenizes and processes the input data.
Load Dataset: The load_dataset function loads the processed dataset.
Fine-Tune Model: The model is fine-tuned using the Hugging Face Trainer class.
Streamlit App (app.py)
User Authentication: Users must log in with a username and email.
Text Generation: Users can input a prompt, and the app generates text using the fine-tuned GPT-2 model.
Logging: User login information is logged and saved in user_log.csv.
Display Logs: Users can view the login information by selecting the "Show Logged User Info" checkbox.

# Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

# License
This project is licensed under the MIT License. See the LICENSE file for more details.
## Contact 
# Linkedin Profile- https://www.linkedin.com/in/krishnaganth-m-468309251/
Mail- krishnaganth2206@gmail.com
